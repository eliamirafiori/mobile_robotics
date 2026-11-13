"""
Decentralized 3D Drone Swarm Area Coverage Simulation (MuJoCo)
==============================================================
Simulates a team of quadrotor drones equipped with downward-facing monocular
or stereo cameras exploring a 3D environment with obstacles.

Key Features:
  - Fully decentralized decision-making: drones only communicate with active
    peers within `comm_radius` and automatically redistribute workload when a
    drone fails or recovers.
  - High-speed vectorized camera frustum projection + MuJoCo C-level raycasting
    (`mj_multiRay` with fallback) for realistic 3D building occlusion.
  - Plugin architecture (`@register_algorithm`) allowing new coverage strategies
    to be added in a single function and selected via `--algo`.
"""

import argparse
import os
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import imageio.v3 as iio
import mujoco
import numpy as np


# =====================================================================
# 1. DATA STRUCTURES: SENSOR, DRONE STATE, AND COVERAGE GRID
# =====================================================================


@dataclass
class CameraSpec:
    """Defines the optical and depth parameters of each drone's downward camera."""

    sensor_type: str = "stereo"  # "mono" (single lens) or "stereo" (overlapping L/R lenses)
    fovy: float = 65.0           # Vertical field of view (degrees)
    aspect: float = 1.33         # Width-to-height sensor aspect ratio (4:3)
    baseline: float = 0.12       # Physical distance between left & right stereo lenses (m)
    max_depth: float = 5.5       # Maximum effective sensing range (m)
    min_depth: float = 0.4       # Minimum blind-spot distance near the lens (m)


@dataclass
class DroneState:
    """Stores the real-time physical and decentralized networking state of one drone."""

    id: int
    pos: np.ndarray              # Current 3D world position [x, y, z] (m)
    vel: np.ndarray              # Current 3D linear velocity [vx, vy, vz] (m/s)
    yaw: float = 0.0             # Heading angle around Z-axis (rad)
    active: bool = True          # False if battery depleted / crashed (triggers swarm self-healing)
    comm_radius: float = 9.0     # Peer-to-peer radio communication range (m)
    target: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))


@dataclass
class CoverageGrid:
    """
    2D discretized ground map tracking:
      1. Individual coverage (`drone_sole_visited`): cells actually seen by each drone's camera.
      2. Decentralized belief (`drone_belief_uncertainty`): each drone's local estimate of
         unexplored areas, updated via its own camera and peer-to-peer map sharing.
      3. Static 3D obstacle mask (`free_cells`): excludes cells inside buildings/walls.
    """

    num_drones: int
    size: float = 16.0           # Square arena side length in meters (-size/2 to +size/2)
    resolution: int = 45         # Grid cells per axis (total cells = resolution^2)

    def __post_init__(self):
        half = self.size / 2.0
        step = half / self.resolution
        coords = np.linspace(-half + step, half - step, self.resolution)
        gx, gy = np.meshgrid(coords, coords)

        # (N_cells, 2) array of [x, y] coordinates for every ground cell center
        self.cell_centers = np.stack([gx.ravel(), gy.ravel()], axis=-1).astype(np.float64)
        self.num_cells = len(self.cell_centers)

        # Precompute 3D ground target points (z = 0.02m) once to speed up raycasting
        self.ground_points_3d = np.column_stack(
            [self.cell_centers, np.full(self.num_cells, 0.02, dtype=np.float64)]
        )

        # Boolean mask of cells physically visited by each individual drone
        self.drone_sole_visited = np.zeros((self.num_drones, self.num_cells), dtype=bool)

        # Uncertainty map per drone: 1.0 = completely unexplored, 0.05 = thoroughly scanned
        self.drone_belief_uncertainty = np.ones((self.num_drones, self.num_cells), dtype=np.float32)

        # Mask of traversable ground cells (False where 3D buildings sit)
        self.free_cells = np.ones(self.num_cells, dtype=bool)
        self.obstacle_centers = np.empty((0, 2), dtype=np.float64)

    def mark_obstacles(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        """Casts vertical rays from above the arena to detect static 3D obstacles."""
        ray_origin = np.array([0.0, 0.0, 10.0], dtype=np.float64)
        ray_dir = np.array([0.0, 0.0, -1.0], dtype=np.float64)
        geom_id_out = np.zeros(1, dtype=np.int32)

        for idx, (cx, cy) in enumerate(self.cell_centers):
            ray_origin[0], ray_origin[1] = cx, cy
            hit_dist = mujoco.mj_ray(model, data, ray_origin, ray_dir, None, 1, -1, geom_id_out)
            # Floor is at z=0 (hit_dist = 10.0m). Anything hit earlier (z > 0.2m) is an obstacle.
            if 0.0 < hit_dist < 9.8:
                self.free_cells[idx] = False
                self.drone_belief_uncertainty[:, idx] = 0.0

        # Cache obstacle coordinates and free cell count for fast algorithm queries
        self.obstacle_centers = self.cell_centers[~self.free_cells]
        self.total_free_cells = max(1, int(np.sum(self.free_cells)))

    def update_from_cameras(
        self,
        drones: List[DroneState],
        drone_body_ids: List[int],
        cam_spec: CameraSpec,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        decay_rate: float = 0.25,
    ) -> None:
        """
        Projects each active drone's camera frustum onto the ground, checks 3D line-of-sight
        occlusion using MuJoCo raycasting, and fuses maps between nearby neighbors.
        """
        tan_y = np.tan(np.deg2rad(cam_spec.fovy * 0.5))
        tan_x = tan_y * cam_spec.aspect
        half_baseline = cam_spec.baseline * 0.5
        geom_id_out = np.zeros(1, dtype=np.int32)

        # --- Step 1: Individual Camera Sensing + 3D Occlusion Check ---
        for d in drones:
            if not d.active:
                continue

            # Vectorized relative geometry from drone to all ground cells
            rel_xy = self.cell_centers - d.pos[:2]
            height = max(float(d.pos[2]), 0.2)
            dists_3d = np.sqrt(np.einsum("ij,ij->i", rel_xy, rel_xy) + height * height)

            # Rotate ground offsets into the drone's local yaw coordinate frame
            cos_y, sin_y = np.cos(-d.yaw), np.sin(-d.yaw)
            local_x = cos_y * rel_xy[:, 0] - sin_y * rel_xy[:, 1]
            local_y = sin_y * rel_xy[:, 0] + cos_y * rel_xy[:, 1]

            # Check camera frustum limits (mono = 1 frustum, stereo = overlapping L/R frustums)
            max_x, max_y = height * tan_x, height * tan_y
            within_range = (
                (np.abs(local_y) <= max_y)
                & (dists_3d <= cam_spec.max_depth)
                & (dists_3d >= cam_spec.min_depth)
                & self.free_cells
            )

            if cam_spec.sensor_type == "mono":
                in_frustum = within_range & (np.abs(local_x) <= max_x)
            else:
                in_left = np.abs(local_x + half_baseline) <= max_x
                in_right = np.abs(local_x - half_baseline) <= max_x
                in_frustum = within_range & in_left & in_right

            candidate_indices = np.where(in_frustum)[0]
            if len(candidate_indices) == 0:
                continue

            # Compute normalized ray directions from camera lens to candidate ground cells
            cam_origin = np.array([d.pos[0], d.pos[1], d.pos[2] - 0.08], dtype=np.float64)
            ray_vecs = self.ground_points_3d[candidate_indices] - cam_origin
            ray_lengths = np.linalg.norm(ray_vecs, axis=1)
            ray_dirs = ray_vecs / ray_lengths[:, None]

            # Batch raycast in C if `mj_multiRay` is available; otherwise fallback to loop
            body_id = drone_body_ids[d.id]
            n_rays = len(candidate_indices)
            if hasattr(mujoco, "mj_multiRay"):
                hit_dists = np.empty(n_rays, dtype=np.float64)
                hit_geomids = np.empty(n_rays, dtype=np.int32)
                mujoco.mj_multiRay(
                    model,
                    data,
                    cam_origin,
                    ray_dirs.ravel(),
                    None,
                    1,
                    body_id,
                    hit_geomids,
                    hit_dists,
                    None,
                    n_rays,
                    cam_spec.max_depth + 1.0,
                )
                unoccluded = (hit_dists < 0.0) | (hit_dists >= (ray_lengths - 0.15))
            else:
                unoccluded = np.zeros(n_rays, dtype=bool)
                for k in range(n_rays):
                    hd = mujoco.mj_ray(model, data, cam_origin, ray_dirs[k], None, 1, body_id, geom_id_out)
                    unoccluded[k] = (hd < 0.0) or (hd >= (ray_lengths[k] - 0.15))

            visible_cells = candidate_indices[unoccluded]
            if len(visible_cells) > 0:
                self.drone_sole_visited[d.id, visible_cells] = True
                self.drone_belief_uncertainty[d.id, visible_cells] = np.maximum(
                    0.05, self.drone_belief_uncertainty[d.id, visible_cells] * (1.0 - decay_rate)
                )

        # --- Step 2: Decentralized Peer-to-Peer Map Merging ---
        # Active drones within `comm_radius` share their uncertainty maps so peers
        # do not re-explore regions already covered by teammates.
        active_ids = [d.id for d in drones if d.active]
        if len(active_ids) > 1:
            positions = np.array([drones[i].pos for i in active_ids])
            for idx_a in range(len(active_ids)):
                i = active_ids[idx_a]
                dists = np.linalg.norm(positions[idx_a + 1 :] - positions[idx_a], axis=1)
                for offset, dist in enumerate(dists):
                    j = active_ids[idx_a + 1 + offset]
                    if dist <= drones[i].comm_radius:
                        merged = np.minimum(
                            self.drone_belief_uncertainty[i],
                            self.drone_belief_uncertainty[j],
                        )
                        self.drone_belief_uncertainty[i] = merged
                        self.drone_belief_uncertainty[j] = merged.copy()

    def get_drone_coverage_pct(self, drone_id: int) -> float:
        """Percentage of traversable area scanned by a single drone."""
        covered = np.sum(self.drone_sole_visited[drone_id] & self.free_cells)
        return float(covered) / self.total_free_cells * 100.0

    @property
    def combined_visited_map(self) -> np.ndarray:
        """Boolean union of all individual drone coverage maps."""
        return np.any(self.drone_sole_visited, axis=0)

    @property
    def combined_coverage_pct(self) -> float:
        """Total percentage of traversable area covered by the entire swarm."""
        covered = np.sum(self.combined_visited_map & self.free_cells)
        return float(covered) / self.total_free_cells * 100.0


# =====================================================================
# 2. SWAPPABLE COVERAGE ALGORITHMS (PLUGIN REGISTRY)
# =====================================================================

AlgorithmFunc = Callable[[DroneState, List[DroneState], CoverageGrid], np.ndarray]
ALGORITHMS: Dict[str, AlgorithmFunc] = {}


def register_algorithm(name: str):
    """
    Decorator to register any coverage planning function by CLI name.

    Signature required for any registered algorithm:
        def my_algo(ego: DroneState, neighbors: List[DroneState], grid: CoverageGrid) -> np.ndarray
    Returns a 3D target coordinate `[target_x, target_y, target_z]`.
    """

    def decorator(func: AlgorithmFunc) -> AlgorithmFunc:
        ALGORITHMS[name] = func
        return func

    return decorator


def apply_obstacle_clearance(
    target_xy: np.ndarray, grid: CoverageGrid, clearance: float = 1.3
) -> np.ndarray:
    """
    Shared helper: pushes a 2D target waypoint away from 3D building footprints
    and clamps it safely inside the arena boundary walls.
    """
    if len(grid.obstacle_centers) > 0:
        obs_vecs = target_xy - grid.obstacle_centers
        obs_dists = np.linalg.norm(obs_vecs, axis=1)
        close_mask = obs_dists < clearance

        if np.any(close_mask):
            weights = (clearance - obs_dists[close_mask]) / (obs_dists[close_mask] + 1e-3)
            push = np.sum(obs_vecs[close_mask] * weights[:, None], axis=0)
            norm_push = np.linalg.norm(push)
            if norm_push > 1e-4:
                target_xy = target_xy + (push / norm_push) * 0.65

    half_limit = (grid.size * 0.5) - 0.8
    return np.clip(target_xy, -half_limit, half_limit)


@register_algorithm("voronoi")
def voronoi_coverage(
    ego: DroneState, neighbors: List[DroneState], grid: CoverageGrid
) -> np.ndarray:
    """
    Algorithm 1: Decentralized Dynamic Voronoi Partitioning + Active Sweep
    -----------------------------------------------------------------------
    - Territory Split: Each drone claims all free cells closer to itself than
      to any active neighbor. If a drone fails (`nb.active == False`), its
      neighbors automatically absorb its territory on the next step.
    - Target Selection: Blends the uncertainty-weighted Voronoi centroid (30%)
      with the highest-priority unvisited cell inside its Voronoi zone (70%)
      so the drone continuously sweeps corners rather than hovering in place.
    """
    pts = grid.cell_centers
    unc = grid.drone_belief_uncertainty[ego.id]

    # 1. Compute ego's Voronoi cell against active neighbors
    ego_dists = np.linalg.norm(pts - ego.pos[:2], axis=1)
    in_ego_cell = grid.free_cells.copy()
    for nb in neighbors:
        if nb.active and nb.id != ego.id:
            nb_dists = np.linalg.norm(pts - nb.pos[:2], axis=1)
            in_ego_cell &= ego_dists <= nb_dists

    if not np.any(in_ego_cell):
        in_ego_cell = grid.free_cells.copy()

    cell_pts = pts[in_ego_cell]
    cell_unc = unc[in_ego_cell]
    cell_dists = ego_dists[in_ego_cell]

    # 2. Uncertainty-weighted centroid (Lloyd's step)
    centroid_xy = np.average(cell_pts, axis=0, weights=(cell_unc**3) + 1e-5)

    # 3. Local greedy sweep point with momentum hysteresis
    dist_to_curr = np.linalg.norm(cell_pts - ego.target[:2], axis=1)
    hysteresis = 0.35 * np.exp(-dist_to_curr / 1.5)
    sweep_scores = (cell_unc**2) / (0.35 * cell_dists + 1.0) + hysteresis
    sweep_xy = cell_pts[int(np.argmax(sweep_scores))]

    target_xy = apply_obstacle_clearance(0.30 * centroid_xy + 0.70 * sweep_xy, grid)
    return np.array([float(target_xy[0]), float(target_xy[1]), 2.4], dtype=np.float64)


@register_algorithm("frontier")
def frontier_coverage(
    ego: DroneState, neighbors: List[DroneState], grid: CoverageGrid
) -> np.ndarray:
    """
    Algorithm 2: Cooperative Information-Gain Frontier Exploration
    --------------------------------------------------------------
    - Evaluates candidate unvisited cells by summing expected camera information
      gain within a 2.0m radius.
    - Penalizes candidates that are closer to a neighbor or near a neighbor's
      current target waypoint so drones naturally spread across distinct regions.
    """
    pts = grid.cell_centers
    unc = grid.drone_belief_uncertainty[ego.id]
    free = grid.free_cells

    unvisited_mask = (unc > 0.25) & free
    if not np.any(unvisited_mask):
        unvisited_mask = free.copy()

    cand_indices = np.where(unvisited_mask)[0]
    if len(cand_indices) > 80:
        cand_indices = cand_indices[np.argsort(unc[cand_indices])[-80:]]

    cand_pts = pts[cand_indices]
    free_pts = pts[free]
    free_unc = unc[free]

    # Expected camera footprint information gain around each candidate
    dist_matrix = np.linalg.norm(cand_pts[:, None, :] - free_pts[None, :, :], axis=2)
    info_gain = np.sum((dist_matrix <= 2.0) * free_unc[None, :], axis=1)

    ego_dists = np.linalg.norm(cand_pts - ego.pos[:2], axis=1)
    momentum = 2.0 * np.exp(-np.linalg.norm(cand_pts - ego.target[:2], axis=1) / 1.5)

    # Repel from active neighbors' positions and declared targets
    peer_penalty = np.zeros(len(cand_pts), dtype=np.float64)
    for nb in neighbors:
        if not nb.active or nb.id == ego.id:
            continue
        d_nb_pos = np.linalg.norm(cand_pts - nb.pos[:2], axis=1)
        d_nb_tgt = np.linalg.norm(cand_pts - nb.target[:2], axis=1)
        peer_penalty += np.where(d_nb_pos < ego_dists, 6.0, 0.0)
        peer_penalty += 12.0 * np.exp(-(d_nb_tgt**2) / 5.0)

    utility = (1.5 * info_gain) - (0.85 * ego_dists) + momentum - peer_penalty
    best_xy = apply_obstacle_clearance(cand_pts[int(np.argmax(utility))], grid)
    return np.array([float(best_xy[0]), float(best_xy[1]), 2.4], dtype=np.float64)


@register_algorithm("spiral_inward")
def spiral_inward_coverage(
    ego: DroneState, neighbors: List[DroneState], grid: CoverageGrid
) -> np.ndarray:
    """
    Algorithm 3: Decentralized Outside-to-Inside Concentric Ring Sweep
    ------------------------------------------------------------------
    - Drones form an evenly spaced ring around the outer perimeter and spiral
      gradually inward toward the center as outer rings get explored.
    - Self-healing: Phase angles around the circle are computed dynamically from
      the sorted list of ACTIVE peer IDs, so if a drone drops out, the remaining
      drones automatically re-space to close the angular gap.
    """
    pts = grid.cell_centers
    unc = grid.drone_belief_uncertainty[ego.id]
    free = grid.free_cells

    # 1. Determine active swarm size and ego's angular slot (self-healing ring)
    active_ids = sorted([ego.id] + [nb.id for nb in neighbors if nb.active])
    n_active = len(active_ids)
    my_slot = active_ids.index(ego.id)

    # 2. Find the outermost unexplored radius across the arena
    radii = np.linalg.norm(pts, axis=1)
    unvisited_mask = (unc > 0.25) & free
    if np.any(unvisited_mask):
        # Target the 90th percentile radius of remaining unexplored cells (outside -> inside)
        target_radius = float(np.percentile(radii[unvisited_mask], 90))
    else:
        # Once 100% visited, patrol mid-ring
        target_radius = grid.size * 0.25

    # 3. Advance a continuous angular sweep based on arena uncertainty progress
    # As overall mean uncertainty drops from 1.0 -> 0.05, the swarm completes ~3 full rotations
    mean_unc = float(np.mean(unc[free]))
    sweep_progress_angle = (1.0 - mean_unc) * (6.0 * np.pi)
    desired_angle = sweep_progress_angle + (2.0 * np.pi * my_slot / n_active)

    # Ideal point on the shrinking spiral
    ideal_xy = np.array(
        [target_radius * np.cos(desired_angle), target_radius * np.sin(desired_angle)],
        dtype=np.float64,
    )

    # Snap to the highest-uncertainty free grid cell near the ideal spiral point
    dist_to_ideal = np.linalg.norm(pts[free] - ideal_xy, axis=1)
    score = (unc[free] * 2.5) - (0.6 * dist_to_ideal)
    chosen_xy = apply_obstacle_clearance(pts[free][int(np.argmax(score))], grid)

    return np.array([float(chosen_xy[0]), float(chosen_xy[1]), 2.4], dtype=np.float64)


# =====================================================================
# 3. 3D SCENARIO & MUJOCO XML SCENE BUILDER
# =====================================================================


def build_scenario_obstacles_xml(
    scenario: str, custom_mesh_path: Optional[str] = None
) -> Tuple[str, str]:
    """Returns `(assets_xml, worldbody_obstacles_xml)` for the selected 3D environment."""
    if scenario == "urban":
        obstacles = """
        <geom name="bldg_1" type="box" pos="-3.5 3.0 1.4" size="1.4 1.4 1.4" rgba="0.35 0.38 0.45 1"/>
        <geom name="bldg_2" type="box" pos="3.5 -3.0 1.6" size="1.5 1.2 1.6" rgba="0.35 0.38 0.45 1"/>
        <geom name="bldg_3" type="box" pos="3.2 3.5 1.2" size="1.2 1.2 1.2" rgba="0.4 0.35 0.35 1"/>
        <geom name="bldg_4" type="box" pos="-3.8 -3.5 1.5" size="1.1 1.6 1.5" rgba="0.35 0.42 0.4 1"/>
        <geom name="central_monument" type="cylinder" pos="0 0 1.1" size="0.9 1.1" rgba="0.45 0.45 0.5 1"/>
        """
        return "", obstacles

    if scenario == "ruins":
        obstacles = """
        <geom name="wall_1" type="box" pos="-2.5 0.0 0.9" size="0.3 3.2 0.9" rgba="0.5 0.42 0.35 1"/>
        <geom name="wall_2" type="box" pos="2.5 1.5 0.8" size="2.2 0.3 0.8" rgba="0.5 0.42 0.35 1"/>
        <geom name="rubble_1" type="box" pos="1.0 -3.0 0.6" size="1.5 1.0 0.6" euler="15 20 35" rgba="0.45 0.4 0.35 1"/>
        <geom name="pillar_1" type="cylinder" pos="-4.5 4.0 1.5" size="0.6 1.5" rgba="0.55 0.5 0.45 1"/>
        <geom name="pillar_2" type="cylinder" pos="4.5 -4.0 1.3" size="0.7 1.3" rgba="0.55 0.5 0.45 1"/>
        """
        return "", obstacles

    if scenario == "custom_mesh" and custom_mesh_path and os.path.exists(custom_mesh_path):
        assets = f'<mesh name="custom_env_mesh" file="{custom_mesh_path}" scale="1 1 1"/>'
        obstacles = '<geom name="custom_3d_map" type="mesh" mesh="custom_env_mesh" pos="0 0 0" rgba="0.5 0.5 0.55 1"/>'
        return assets, obstacles

    return "", ""


def build_swarm_xml(
    num_drones: int,
    cam_spec: CameraSpec,
    scenario: str = "urban",
    custom_mesh_path: Optional[str] = None,
    arena_size: float = 16.0,
) -> str:
    """Constructs the complete MuJoCo MJCF XML model with drones, cameras, and 3D obstacles."""
    half = arena_size * 0.5
    assets_xml, obstacles_xml = build_scenario_obstacles_xml(scenario, custom_mesh_path)

    drone_bodies = []
    actuators = []
    palette = [
        "0.2 0.85 0.35 1",
        "0.2 0.65 1.0 1",
        "1.0 0.6 0.15 1",
        "0.95 0.25 0.8 1",
        "0.15 0.9 0.9 1",
        "0.95 0.9 0.2 1",
    ]

    for i in range(num_drones):
        angle = 2.0 * np.pi * i / num_drones
        sx, sy = 6.2 * np.cos(angle), 6.2 * np.sin(angle)
        color = palette[i % len(palette)]

        if cam_spec.sensor_type == "mono":
            cameras_xml = f"""
            <camera name="cam_mono_{i}" pos="0 0 -0.06" euler="0 0 0" fovy="{cam_spec.fovy}"/>
            <geom name="lens_mono_{i}" type="cylinder" pos="0 0 -0.05" size="0.04 0.02" rgba="0.1 0.1 0.1 1"/>
            """
        else:
            hb = cam_spec.baseline * 0.5
            cameras_xml = f"""
            <camera name="cam_left_{i}" pos="{-hb:.3f} 0 -0.06" euler="0 0 0" fovy="{cam_spec.fovy}"/>
            <camera name="cam_right_{i}" pos="{hb:.3f} 0 -0.06" euler="0 0 0" fovy="{cam_spec.fovy}"/>
            <geom name="lens_l_{i}" type="cylinder" pos="{-hb:.3f} 0 -0.05" size="0.035 0.02" rgba="0.1 0.1 0.1 1"/>
            <geom name="lens_r_{i}" type="cylinder" pos="{hb:.3f} 0 -0.05" size="0.035 0.02" rgba="0.1 0.1 0.1 1"/>
            """

        drone_bodies.append(f"""
        <body name="drone_{i}" pos="{sx:.2f} {sy:.2f} 2.4">
            <freejoint name="joint_{i}"/>
            <geom name="core_{i}" type="box" size="0.22 0.22 0.06" rgba="{color}" mass="1.0"/>
            <geom type="cylinder" pos="0.22 0.22 0.04" size="0.12 0.01" rgba="0.8 0.8 0.8 0.7" mass="0.05"/>
            <geom type="cylinder" pos="-0.22 0.22 0.04" size="0.12 0.01" rgba="0.8 0.8 0.8 0.7" mass="0.05"/>
            <geom type="cylinder" pos="0.22 -0.22 0.04" size="0.12 0.01" rgba="0.8 0.8 0.8 0.7" mass="0.05"/>
            <geom type="cylinder" pos="-0.22 -0.22 0.04" size="0.12 0.01" rgba="0.8 0.8 0.8 0.7" mass="0.05"/>
            {cameras_xml}
        </body>
        <body name="target_vis_{i}" mocap="true" pos="{sx:.2f} {sy:.2f} 0.05">
            <geom name="target_geom_{i}" type="sphere" size="0.16" rgba="{color}" contype="0" conaffinity="0"/>
        </body>
        """)

        actuators.append(f"""
        <general name="fx_{i}" joint="joint_{i}" gear="1 0 0 0 0 0" ctrlrange="-25 25"/>
        <general name="fy_{i}" joint="joint_{i}" gear="0 1 0 0 0 0" ctrlrange="-25 25"/>
        <general name="fz_{i}" joint="joint_{i}" gear="0 0 1 0 0 0" ctrlrange="-25 35"/>
        """)

    return f"""
    <mujoco model="drone_swarm_3d_coverage">
        <option timestep="0.01" gravity="0 0 -9.81" integrator="RK4"/>
        <asset>
            <texture name="grid_tex" type="2d" builtin="checker" width="512" height="512"
                     rgb1="0.18 0.20 0.24" rgb2="0.23 0.26 0.30"/>
            <material name="floor_mat" texture="grid_tex" texrepeat="8 8" reflectance="0.1"/>
            {assets_xml}
        </asset>
        <worldbody>
            <light pos="0 0 18" dir="0 0 -1" directional="true" diffuse="0.8 0.8 0.8"/>
            <geom name="floor" type="plane" size="{half} {half} 0.1" material="floor_mat"/>
            <geom type="box" pos="0 {half} 0.6" size="{half} 0.1 0.6" rgba="0.4 0.4 0.4 0.4"/>
            <geom type="box" pos="0 {-half} 0.6" size="{half} 0.1 0.6" rgba="0.4 0.4 0.4 0.4"/>
            <geom type="box" pos="{half} 0 0.6" size="0.1 {half} 0.6" rgba="0.4 0.4 0.4 0.4"/>
            <geom type="box" pos="{-half} 0 0.6" size="0.1 {half} 0.6" rgba="0.4 0.4 0.4 0.4"/>
            {obstacles_xml}
            {"".join(drone_bodies)}
        </worldbody>
        <actuator>
            {"".join(actuators)}
        </actuator>
    </mujoco>
    """


# =====================================================================
# 4. REPORTING: COVERAGE COMPARISON PLOTS & ONBOARD CAMERA SNAPSHOTS
# =====================================================================


def save_coverage_and_camera_reports(
    grid: CoverageGrid,
    drones: List[DroneState],
    cam_spec: CameraSpec,
    model: mujoco.MjModel,
    data: mujoco.MjData,
) -> None:
    """Exports individual/combined coverage maps and onboard camera images."""
    res = grid.resolution
    half = grid.size * 0.5
    obstacle_mask_2d = (~grid.free_cells).reshape(res, res)

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        num_d = len(drones)
        fig, axes = plt.subplots(1, num_d + 1, figsize=(3.6 * (num_d + 1), 3.8), dpi=140)

        for i in range(num_d):
            ax = axes[i]
            cov_2d = grid.drone_sole_visited[i].reshape(res, res).astype(float)
            cov_2d[obstacle_mask_2d] = -0.5
            ax.imshow(
                cov_2d,
                cmap="viridis",
                vmin=-0.5,
                vmax=1.0,
                origin="lower",
                extent=[-half, half, -half, half],
            )
            ax.plot(drones[i].pos[0], drones[i].pos[1], "ro", markersize=7)
            ax.set_title(f"Drone {i} Only\n({grid.get_drone_coverage_pct(i):.1f}% covered)")
            ax.set_xlabel("X (m)")
            if i == 0:
                ax.set_ylabel("Y (m)")

        ax_comb = axes[-1]
        comb_2d = grid.combined_visited_map.reshape(res, res).astype(float)
        comb_2d[obstacle_mask_2d] = -0.5
        ax_comb.imshow(
            comb_2d,
            cmap="plasma",
            vmin=-0.5,
            vmax=1.0,
            origin="lower",
            extent=[-half, half, -half, half],
        )
        for d in drones:
            ax_comb.plot(d.pos[0], d.pos[1], "wo", markersize=6)
        ax_comb.set_title(f"COMBINED SWARM\n({grid.combined_coverage_pct:.1f}% total)")
        ax_comb.set_xlabel("X (m)")

        plt.tight_layout()
        plt.savefig("coverage_maps_comparison.png")
        plt.close(fig)
        print("-> Saved individual + combined coverage maps to 'coverage_maps_comparison.png'")
    except ImportError:
        print("-> (Install matplotlib to generate 'coverage_maps_comparison.png')")

    renderer = mujoco.Renderer(model, height=320, width=420)
    if cam_spec.sensor_type == "mono":
        renderer.update_scene(data, camera="cam_mono_0")
        iio.imwrite("drone0_mono_camera.png", renderer.render())
        print("-> Saved Drone 0 monocular camera frame to 'drone0_mono_camera.png'")
    else:
        renderer.update_scene(data, camera="cam_left_0")
        left_rgb = renderer.render()
        renderer.update_scene(data, camera="cam_right_0")
        right_rgb = renderer.render()
        iio.imwrite("drone0_stereo_pair.png", np.concatenate([left_rgb, right_rgb], axis=1))
        print("-> Saved Drone 0 left+right stereo camera pair to 'drone0_stereo_pair.png'")
    renderer.close()


# =====================================================================
# 5. MAIN SIMULATION LOOP & CONTROLLER
# =====================================================================


def run_simulation(
    algo_name: str,
    mode: str,
    sensor_type: str,
    scenario: str,
    custom_mesh: Optional[str],
    num_drones: int,
    max_steps: int,
) -> None:
    cam_spec = CameraSpec(sensor_type=sensor_type)
    algo_fn = ALGORITHMS[algo_name]

    xml_str = build_swarm_xml(
        num_drones, cam_spec, scenario=scenario, custom_mesh_path=custom_mesh
    )
    model = mujoco.MjModel.from_xml_string(xml_str)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    # Pre-cache MuJoCo body & geom IDs once (avoids slow string lookups in the physics loop)
    drone_body_ids = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"drone_{i}")
        for i in range(num_drones)
    ]
    drone_core_geom_ids = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"core_{i}")
        for i in range(num_drones)
    ]

    grid = CoverageGrid(num_drones=num_drones, size=16.0, resolution=45)
    grid.mark_obstacles(model, data)

    drones = [
        DroneState(
            id=i,
            pos=data.qpos[i * 7 : i * 7 + 3].copy(),
            vel=np.zeros(3, dtype=np.float64),
            target=data.qpos[i * 7 : i * 7 + 3].copy(),
        )
        for i in range(num_drones)
    ]

    print(
        f"=== Swarm Sim | Mode: {mode.upper()} | Algo: {algo_name.upper()} | "
        f"Sensor: {sensor_type.upper()} | Scenario: {scenario.upper()} ==="
    )

    # PD flight controller gains and quadrotor mass (kg)
    kp, kd, mass = 6.0, 4.5, 1.2
    gravity_force = mass * 9.81

    def step_simulation(step_idx: int) -> None:
        # 1. Sync drone state from MuJoCo & stabilize attitude (upright hover)
        for i, d in enumerate(drones):
            qpos_adr, qvel_adr = i * 7, i * 6
            d.pos[:] = data.qpos[qpos_adr : qpos_adr + 3]
            d.vel[:] = data.qvel[qvel_adr : qvel_adr + 3]
            data.qpos[qpos_adr + 3 : qpos_adr + 7] = (1.0, 0.0, 0.0, 0.0)
            data.qvel[qvel_adr + 3 : qvel_adr + 6] = (0.0, 0.0, 0.0)

        # 2. Inject dynamic failure event at step 500 to verify swarm self-healing
        if step_idx == 500 and len(drones) > 1:
            drones[0].active = False
            model.geom_rgba[drone_core_geom_ids[0]] = [0.8, 0.1, 0.1, 0.4]
            print(f"\n[EVENT @ Step {step_idx}] Drone 0 FAILED! Surviving drones redistributing area...")

        # 3. Sensor update + P2P map fusion at 25 Hz (every 4 steps)
        if step_idx % 4 == 0:
            grid.update_from_cameras(drones, drone_body_ids, cam_spec, model, data)

        # 4. Decentralized target replanning at 10 Hz (every 10 steps)
        if step_idx % 10 == 0:
            all_positions = np.array([d.pos for d in drones])
            for i, d in enumerate(drones):
                if not d.active:
                    d.target[:] = (d.pos[0], d.pos[1], 0.15)
                    continue

                # Filter neighbors within radio communication range
                dists = np.linalg.norm(all_positions - d.pos, axis=1)
                local_neighbors = [
                    drones[j]
                    for j in range(num_drones)
                    if j != i and dists[j] <= d.comm_radius
                ]

                new_target = algo_fn(d, local_neighbors, grid)
                if new_target is not None:
                    d.target[:] = new_target
                else:
                    d.target[:] = (d.pos[0], d.pos[1], 2.4)

                data.mocap_pos[i] = (d.target[0], d.target[1], 0.05)

        # 5. Low-level PD position control + gravity compensation
        for i, d in enumerate(drones):
            vel_cmd = np.clip((d.target - d.pos) * kp, -7.0, 7.0)
            acc_cmd = vel_cmd - kd * d.vel
            ctrl_adr = i * 3
            data.ctrl[ctrl_adr] = mass * acc_cmd[0]
            data.ctrl[ctrl_adr + 1] = mass * acc_cmd[1]
            data.ctrl[ctrl_adr + 2] = mass * acc_cmd[2] + gravity_force

        mujoco.mj_step(model, data)

        if step_idx % 250 == 0:
            indiv_str = " | ".join(
                f"D{i}:{grid.get_drone_coverage_pct(i):.0f}%" for i in range(num_drones)
            )
            print(
                f"Step {step_idx:04d} | Combined: {grid.combined_coverage_pct:5.1f}% | "
                f"Individual -> [{indiv_str}]"
            )

    if mode == "visual":
        from mujoco import viewer

        with viewer.launch_passive(model, data) as v:
            step_idx = 0
            while v.is_running() and step_idx < max_steps:
                t0 = time.perf_counter()
                step_simulation(step_idx)
                v.sync()
                step_idx += 1
                elapsed = time.perf_counter() - t0
                if elapsed < model.opt.timestep:
                    time.sleep(model.opt.timestep - elapsed)
    else:
        for step_idx in range(max_steps):
            step_simulation(step_idx)

    save_coverage_and_camera_reports(grid, drones, cam_spec, model, data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="3D Camera-Based Drone Swarm Coverage")
    parser.add_argument("--algo", type=str, default="voronoi", choices=list(ALGORITHMS.keys()))
    parser.add_argument("--mode", type=str, default="headless", choices=["headless", "visual"])
    parser.add_argument("--sensor", type=str, default="stereo", choices=["mono", "stereo"])
    parser.add_argument(
        "--scenario",
        type=str,
        default="flat",
        choices=["flat", "urban", "ruins", "custom_mesh"],
    )
    parser.add_argument("--mesh-file", type=str, default=None, help="Path to custom .stl or .obj mesh")
    parser.add_argument("--drones", type=int, default=4)
    parser.add_argument("--steps", type=int, default=1200)
    args = parser.parse_args()

    run_simulation(
        algo_name=args.algo,
        mode=args.mode,
        sensor_type=args.sensor,
        scenario=args.scenario,
        custom_mesh=args.mesh_file,
        num_drones=args.drones,
        max_steps=args.steps,
    )

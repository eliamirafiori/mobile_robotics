import argparse
import os
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional
import numpy as np
import mujoco
import imageio.v3 as iio

# =====================================================================
# 1. SENSOR & SCENARIO CONFIGURATION
# =====================================================================


@dataclass
class CameraSpec:
    sensor_type: str = "stereo"  # "mono" or "stereo"
    fovy: float = 65.0  # Vertical FOV in degrees
    aspect: float = 1.33  # 4:3 sensor aspect ratio
    baseline: float = 0.12  # 12 cm stereo baseline (distance between L/R cameras)
    max_depth: float = 5.5  # Max reliable depth / sensing range (meters)
    min_depth: float = 0.4  # Minimum depth range (meters)


@dataclass
class CoverageGrid:
    """Grid representation that tracks BOTH global truth and per-drone individual maps."""

    num_drones: int
    size: float = 16.0
    resolution: int = 45

    def __post_init__(self):
        half = self.size / 2.0
        coords = np.linspace(
            -half + (half / self.resolution),
            half - (half / self.resolution),
            self.resolution,
        )
        gx, gy = np.meshgrid(coords, coords)
        # 3D ground coordinates (x, y, z=0.02)
        self.cell_centers = np.stack([gx.ravel(), gy.ravel()], axis=-1)
        self.num_cells = len(self.cell_centers)

        # Individual coverage maps (solely what each drone's own camera captured)
        self.drone_sole_visited = np.zeros(
            (self.num_drones, self.num_cells), dtype=bool
        )

        # Decentralized belief maps (what each drone knows after P2P sharing with neighbors)
        self.drone_belief_uncertainty = np.ones(
            (self.num_drones, self.num_cells), dtype=np.float32
        )

        # Obstacle mask (cells inside 3D buildings are excluded from coverage metric)
        self.free_cells = np.ones(self.num_cells, dtype=bool)

    def mark_obstacles(self, model: mujoco.MjModel, data: mujoco.MjData):
        """Identify grid cells occupied by static 3D obstacles using downward raycasts."""
        pnt = np.array([0.0, 0.0, 10.0], dtype=np.float64)
        vec = np.array([0.0, 0.0, -1.0], dtype=np.float64)
        geomid = np.zeros(1, dtype=np.int32)
        for idx, (cx, cy) in enumerate(self.cell_centers):
            pnt[0], pnt[1] = cx, cy
            dist = mujoco.mj_ray(model, data, pnt, vec, None, 1, -1, geomid)
            # If ray hits an obstacle higher than the floor (z > 0.2m)
            if 0.0 < dist < 9.8:
                self.free_cells[idx] = False
                self.drone_belief_uncertainty[:, idx] = 0.0

    def update_from_cameras(
        self,
        drones: List["DroneState"],
        cam_spec: CameraSpec,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        decay_rate: float = 0.25,
    ):
        """
        Updates each drone's individual coverage using a camera frustum model +
        3D MuJoCo raycasting (mj_ray) for obstacle occlusion.
        """
        half_fovy_rad = np.deg2rad(cam_spec.fovy / 2.0)
        tan_y = np.tan(half_fovy_rad)
        tan_x = tan_y * cam_spec.aspect

        geomid = np.zeros(1, dtype=np.int32)

        for d in drones:
            if not d.active:
                continue

            # Drone body ID to exclude self-collision during raycasting
            body_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_BODY, f"drone_{d.id}"
            )

            # Relative vector from camera to all free ground cells
            rel_xy = self.cell_centers - d.pos[:2]
            height = max(d.pos[2], 0.2)
            dists_3d = np.sqrt(np.sum(rel_xy**2, axis=1) + height**2)

            # Rotate relative XY into drone's yaw frame
            c, s = np.cos(-d.yaw), np.sin(-d.yaw)
            local_x = c * rel_xy[:, 0] - s * rel_xy[:, 1]
            local_y = s * rel_xy[:, 0] + c * rel_xy[:, 1]

            # Check camera Rectangular Frustum bounds
            if cam_spec.sensor_type == "mono":
                in_frustum = (
                    (np.abs(local_x) <= height * tan_x)
                    & (np.abs(local_y) <= height * tan_y)
                    & (dists_3d <= cam_spec.max_depth)
                    & (dists_3d >= cam_spec.min_depth)
                    & self.free_cells
                )
            else:
                # Stereo requires point to be visible inside the overlapping left & right frustums
                half_b = cam_spec.baseline / 2.0
                in_left = (np.abs(local_x + half_b) <= height * tan_x) & (
                    np.abs(local_y) <= height * tan_y
                )
                in_right = (np.abs(local_x - half_b) <= height * tan_x) & (
                    np.abs(local_y) <= height * tan_y
                )
                in_frustum = (
                    in_left
                    & in_right
                    & (dists_3d <= cam_spec.max_depth)
                    & (dists_3d >= cam_spec.min_depth)
                    & self.free_cells
                )

            candidate_indices = np.where(in_frustum)[0]
            cam_origin = np.array(
                [d.pos[0], d.pos[1], d.pos[2] - 0.08], dtype=np.float64
            )

            for idx in candidate_indices:
                target_pt = np.array(
                    [self.cell_centers[idx, 0], self.cell_centers[idx, 1], 0.02],
                    dtype=np.float64,
                )
                ray_vec = target_pt - cam_origin
                ray_len = np.linalg.norm(ray_vec)
                ray_dir = ray_vec / ray_len

                # Raycast to check if any 3D obstacle blocks the camera line of sight
                hit_dist = mujoco.mj_ray(
                    model, data, cam_origin, ray_dir, None, 1, body_id, geomid
                )
                if hit_dist < 0 or hit_dist >= (ray_len - 0.15):
                    # Unoccluded view of ground cell!
                    self.drone_sole_visited[d.id, idx] = True
                    self.drone_belief_uncertainty[d.id, idx] = max(
                        0.05,
                        self.drone_belief_uncertainty[d.id, idx] * (1.0 - decay_rate),
                    )

        # Decentralized P2P Map Merging between drones within comm_radius
        for i, d_i in enumerate(drones):
            if not d_i.active:
                continue
            for j, d_j in enumerate(drones):
                if i < j and d_j.active:
                    if np.linalg.norm(d_i.pos - d_j.pos) <= d_i.comm_radius:
                        merged_unc = np.minimum(
                            self.drone_belief_uncertainty[i],
                            self.drone_belief_uncertainty[j],
                        )
                        self.drone_belief_uncertainty[i] = merged_unc.copy()
                        self.drone_belief_uncertainty[j] = merged_unc.copy()

    def get_drone_coverage_pct(self, drone_id: int) -> float:
        free_count = np.sum(self.free_cells)
        return (
            float(np.sum(self.drone_sole_visited[drone_id] & self.free_cells))
            / max(1, free_count)
            * 100.0
        )

    @property
    def combined_visited_map(self) -> np.ndarray:
        """Union of all individual drone coverage maps."""
        return np.any(self.drone_sole_visited, axis=0)

    @property
    def combined_coverage_pct(self) -> float:
        free_count = np.sum(self.free_cells)
        return (
            float(np.sum(self.combined_visited_map & self.free_cells))
            / max(1, free_count)
            * 100.0
        )


@dataclass
class DroneState:
    id: int
    pos: np.ndarray
    vel: np.ndarray
    yaw: float = 0.0
    active: bool = True
    comm_radius: float = 9.0
    target: np.ndarray = field(default_factory=lambda: np.zeros(3))


# =====================================================================
# 2. SWAPPABLE COVERAGE ALGORITHMS (VORONOI & FRONTIER)
# =====================================================================

AlgorithmFunc = Callable[[DroneState, List[DroneState], CoverageGrid], np.ndarray]
ALGORITHMS: Dict[str, AlgorithmFunc] = {}


def register_algorithm(name: str):
    def decorator(func: AlgorithmFunc):
        ALGORITHMS[name] = func
        return func

    return decorator


def apply_obstacle_clearance(
    target_xy: np.ndarray, grid: CoverageGrid, clearance: float = 1.3
) -> np.ndarray:
    """Pushes a 2D target point away from any 3D building/obstacle footprint."""
    obstacle_pts = grid.cell_centers[~grid.free_cells]
    if len(obstacle_pts) == 0:
        return target_xy

    obs_vecs = target_xy - obstacle_pts
    obs_dists = np.linalg.norm(obs_vecs, axis=1)
    close_mask = obs_dists < clearance

    if np.any(close_mask):
        weights = (clearance - obs_dists[close_mask]) / (obs_dists[close_mask] + 1e-3)
        push = np.sum(obs_vecs[close_mask] * weights[:, None], axis=0)
        norm_push = np.linalg.norm(push)
        if norm_push > 1e-4:
            target_xy = target_xy + (push / norm_push) * 0.65

    half = (grid.size / 2.0) - 0.8
    return np.clip(target_xy, -half, half)


@register_algorithm("voronoi")
def voronoi_coverage(
    ego: DroneState, neighbors: List[DroneState], grid: CoverageGrid
) -> np.ndarray:
    """
    Decentralized Dynamic Voronoi Partitioning + Hybrid Centroid/Sweep:
    1. Partition: Claims all free grid cells closer to `ego` than to any ACTIVE neighbor.
       (When a drone fails, `nb.active` is False, so survivors automatically absorb its cells.)
    2. Hybrid Target: Blends the uncertainty-weighted Voronoi centroid (global balance) with the
       highest-utility unvisited cell inside the Voronoi region (prevents getting stuck in the middle
       between two equal piles of unvisited cells).
    """
    pts = grid.cell_centers
    unc = grid.drone_belief_uncertainty[ego.id]

    # --- 1. Construct Ego Drone's Voronoi Cell ---
    ego_dists = np.linalg.norm(pts - ego.pos[:2], axis=1)
    in_ego_cell = grid.free_cells.copy()

    for nb in neighbors:
        if not nb.active or nb.id == ego.id:
            continue
        nb_dists = np.linalg.norm(pts - nb.pos[:2], axis=1)
        in_ego_cell &= ego_dists <= nb_dists

    # Fallback if cell is empty
    if not np.any(in_ego_cell):
        in_ego_cell = grid.free_cells.copy()

    cell_indices = np.where(in_ego_cell)[0]
    cell_pts = pts[cell_indices]
    cell_unc = unc[cell_indices]
    cell_dists = ego_dists[cell_indices]

    # --- 2. Compute Uncertainty-Weighted Voronoi Centroid (Lloyd's step) ---
    weights = (cell_unc**3) + 1e-5
    voronoi_centroid = np.average(cell_pts, axis=0, weights=weights)

    # --- 3. Select Local Sweep Waypoint Inside Voronoi Cell ---
    # Prefer unvisited cells (high uncertainty) that are reasonably close and aligned with current target
    dist_to_curr_tgt = np.linalg.norm(cell_pts - ego.target[:2], axis=1)
    hysteresis_bonus = 0.35 * np.exp(-dist_to_curr_tgt / 1.5)

    sweep_scores = (cell_unc**2) / (0.35 * cell_dists + 1.0) + hysteresis_bonus
    best_sweep_pt = cell_pts[int(np.argmax(sweep_scores))]

    # Blend 30% Voronoi Centroid + 70% Active Sweep Point so it sweeps every corner of its territory
    target_xy = 0.30 * voronoi_centroid + 0.70 * best_sweep_pt
    target_xy = apply_obstacle_clearance(target_xy, grid)

    return np.array([float(target_xy[0]), float(target_xy[1]), 2.4], dtype=np.float64)


@register_algorithm("frontier")
def frontier_coverage(
    ego: DroneState, neighbors: List[DroneState], grid: CoverageGrid
) -> np.ndarray:
    """
    Decentralized Cooperative Frontier Exploration (Information Gain vs. Cost + Peer Deconfliction):
    1. Identifies boundary ('frontier') cells between covered space and uncovered space.
    2. Evaluates expected camera footprint Information Gain around each frontier candidate.
    3. Subtracts travel distance cost and applies a strong repulsive penalty around active neighbors'
       positions and declared targets so drones spread out across distinct frontiers.
    """
    pts = grid.cell_centers
    unc = grid.drone_belief_uncertainty[ego.id]
    free = grid.free_cells

    # --- 1. Identify Frontier / Unvisited Candidate Cells ---
    unvisited_mask = (unc > 0.25) & free
    if not np.any(unvisited_mask):
        # All cells visited -> patrol cells with highest residual uncertainty
        unvisited_mask = free.copy()

    cand_indices = np.where(unvisited_mask)[0]
    # Subsample up to 80 top uncertain candidates for fast real-time evaluation
    if len(cand_indices) > 80:
        top_order = np.argsort(unc[cand_indices])[-80:]
        cand_indices = cand_indices[top_order]

    cand_pts = pts[cand_indices]

    # --- 2. Compute Information Gain (sum of uncertainty within camera radius ~2.0m) ---
    # Pairwise distance matrix between candidates (M, 2) and all free cells (N, 2)
    free_pts = pts[free]
    free_unc = unc[free]
    diff_matrix = cand_pts[:, None, :] - free_pts[None, :, :]
    dist_matrix = np.linalg.norm(diff_matrix, axis=2)
    info_gain = np.sum((dist_matrix <= 2.0) * free_unc[None, :], axis=1)

    # --- 3. Compute Travel Cost & Momentum ---
    ego_dists = np.linalg.norm(cand_pts - ego.pos[:2], axis=1)
    dist_to_curr_tgt = np.linalg.norm(cand_pts - ego.target[:2], axis=1)
    momentum_bonus = 2.0 * np.exp(-dist_to_curr_tgt / 1.5)

    # --- 4. Decentralized Neighbor Deconfliction ---
    peer_penalty = np.zeros(len(cand_pts), dtype=np.float64)
    for nb in neighbors:
        if not nb.active or nb.id == ego.id:
            continue
        d_nb_pos = np.linalg.norm(cand_pts - nb.pos[:2], axis=1)
        d_nb_tgt = np.linalg.norm(cand_pts - nb.target[:2], axis=1)

        # Penalize candidates that are closer to a neighbor than to ego
        peer_penalty += np.where(d_nb_pos < ego_dists, 6.0, 0.0)
        # Strong Gaussian repulsion around neighbor's current target
        peer_penalty += 12.0 * np.exp(-(d_nb_tgt**2) / 5.0)

    # --- 5. Maximize Total Frontier Utility ---
    utility = (1.5 * info_gain) - (0.85 * ego_dists) + momentum_bonus - peer_penalty
    best_xy = cand_pts[int(np.argmax(utility))]
    best_xy = apply_obstacle_clearance(best_xy, grid)

    return np.array([float(best_xy[0]), float(best_xy[1]), 2.4], dtype=np.float64)


# =====================================================================
# 3. 3D SCENARIO & CAMERA XML BUILDER
# =====================================================================


def build_scenario_obstacles_xml(
    scenario: str, custom_mesh_path: Optional[str] = None
) -> tuple[str, str]:
    """Returns (assets_xml, worldbody_obstacles_xml) for different 3D environments."""
    assets = ""
    obstacles = ""

    if scenario == "urban":
        # 3D Urban Canyon: Buildings, pillars, and walls that cast sensor occlusions
        obstacles = """
        <geom name="bldg_1" type="box" pos="-3.5 3.0 1.4" size="1.4 1.4 1.4" rgba="0.35 0.38 0.45 1"/>
        <geom name="bldg_2" type="box" pos="3.5 -3.0 1.6" size="1.5 1.2 1.6" rgba="0.35 0.38 0.45 1"/>
        <geom name="bldg_3" type="box" pos="3.2 3.5 1.2" size="1.2 1.2 1.2" rgba="0.4 0.35 0.35 1"/>
        <geom name="bldg_4" type="box" pos="-3.8 -3.5 1.5" size="1.1 1.6 1.5" rgba="0.35 0.42 0.4 1"/>
        <geom name="central_monument" type="cylinder" pos="0 0 1.1" size="0.9 1.1" rgba="0.45 0.45 0.5 1"/>
        """
    elif scenario == "ruins":
        # Disaster / Collapsed Structure Scenario: Uneven debris + tilted slabs + pillars
        obstacles = """
        <geom name="wall_1" type="box" pos="-2.5 0.0 0.9" size="0.3 3.2 0.9" rgba="0.5 0.42 0.35 1"/>
        <geom name="wall_2" type="box" pos="2.5 1.5 0.8" size="2.2 0.3 0.8" rgba="0.5 0.42 0.35 1"/>
        <geom name="rubble_1" type="box" pos="1.0 -3.0 0.6" size="1.5 1.0 0.6" euler="15 20 35" rgba="0.45 0.4 0.35 1"/>
        <geom name="pillar_1" type="cylinder" pos="-4.5 4.0 1.5" size="0.6 1.5" rgba="0.55 0.5 0.45 1"/>
        <geom name="pillar_2" type="cylinder" pos="4.5 -4.0 1.3" size="0.7 1.3" rgba="0.55 0.5 0.45 1"/>
        """
    elif (
        scenario == "custom_mesh"
        and custom_mesh_path
        and os.path.exists(custom_mesh_path)
    ):
        # Load any external .stl or .obj 3D environment mesh
        assets = (
            f'<mesh name="custom_env_mesh" file="{custom_mesh_path}" scale="1 1 1"/>'
        )
        obstacles = '<geom name="custom_3d_map" type="mesh" mesh="custom_env_mesh" pos="0 0 0" rgba="0.5 0.5 0.55 1"/>'

    return assets, obstacles


def build_swarm_xml(
    num_drones: int,
    cam_spec: CameraSpec,
    scenario: str = "urban",
    custom_mesh_path: Optional[str] = None,
    arena_size: float = 16.0,
) -> str:
    half = arena_size / 2.0
    assets_xml, obstacles_xml = build_scenario_obstacles_xml(scenario, custom_mesh_path)

    drone_bodies = []
    actuators = []
    colors = [
        "0.2 0.85 0.35 1",
        "0.2 0.65 1.0 1",
        "1.0 0.6 0.15 1",
        "0.95 0.25 0.8 1",
        "0.15 0.9 0.9 1",
        "0.95 0.9 0.2 1",
    ]

    for i in range(num_drones):
        angle = 2 * np.pi * i / num_drones
        sx, sy = 5.5 * np.cos(angle), 5.5 * np.sin(angle)
        color = colors[i % len(colors)]

        # Mount either a single Monocular camera or a Stereo camera pair on the drone body
        if cam_spec.sensor_type == "mono":
            cameras_xml = f"""
            <camera name="cam_mono_{i}" pos="0 0 -0.06" euler="0 0 0" fovy="{cam_spec.fovy}"/>
            <geom name="lens_mono_{i}" type="cylinder" pos="0 0 -0.05" size="0.04 0.02" rgba="0.1 0.1 0.1 1"/>
            """
        else:
            hb = cam_spec.baseline / 2.0
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
            <texture name="grid_tex" type="2d" builtin="checker" width="512" height="512" rgb1="0.18 0.20 0.24" rgb2="0.23 0.26 0.30"/>
            <material name="floor_mat" texture="grid_tex" texrepeat="8 8" reflectance="0.1"/>
            {assets_xml}
        </asset>
        <worldbody>
            <light pos="0 0 18" dir="0 0 -1" directional="true" diffuse="0.8 0.8 0.8"/>
            <geom name="floor" type="plane" size="{half} {half} 0.1" material="floor_mat"/>
            <!-- Boundary Walls -->
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
# 4. EXPORTING INDIVIDUAL & COMBINED COVERAGE MAPS + CAMERA FEEDS
# =====================================================================


def save_coverage_and_camera_reports(
    grid: CoverageGrid,
    drones: List[DroneState],
    cam_spec: CameraSpec,
    model: mujoco.MjModel,
    data: mujoco.MjData,
):
    """
    1. Saves an image showing each drone's individual covered area + the merged global area.
    2. Renders and saves actual onboard Monocular or Stereo RGB+Depth camera views from Drone 0.
    """
    res = grid.resolution
    obstacle_mask_2d = (~grid.free_cells).reshape(res, res)

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        num_d = len(drones)
        fig, axes = plt.subplots(
            1, num_d + 1, figsize=(3.6 * (num_d + 1), 3.8), dpi=140
        )

        # Plot each drone's individual coverage map
        for i in range(num_d):
            ax = axes[i]
            cov_2d = grid.drone_sole_visited[i].reshape(res, res).astype(float)
            cov_2d[obstacle_mask_2d] = -0.5  # Gray/dark for 3D obstacles
            ax.imshow(
                cov_2d,
                cmap="viridis",
                vmin=-0.5,
                vmax=1.0,
                origin="lower",
                extent=[-8, 8, -8, 8],
            )
            ax.plot(
                drones[i].pos[0],
                drones[i].pos[1],
                "ro",
                markersize=7,
                label="Final Pos",
            )
            ax.set_title(
                f"Drone {i} Only\n({grid.get_drone_coverage_pct(i):.1f}% covered)"
            )
            ax.set_xlabel("X (m)")
            if i == 0:
                ax.set_ylabel("Y (m)")

        # Plot Combined / Merged Swarm Map
        ax_comb = axes[-1]
        comb_2d = grid.combined_visited_map.reshape(res, res).astype(float)
        comb_2d[obstacle_mask_2d] = -0.5
        ax_comb.imshow(
            comb_2d,
            cmap="plasma",
            vmin=-0.5,
            vmax=1.0,
            origin="lower",
            extent=[-8, 8, -8, 8],
        )
        for d in drones:
            ax_comb.plot(d.pos[0], d.pos[1], "wo", markersize=6)
        ax_comb.set_title(f"COMBINED SWARM\n({grid.combined_coverage_pct:.1f}% total)")
        ax_comb.set_xlabel("X (m)")

        plt.tight_layout()
        plt.savefig("coverage_maps_comparison.png")
        plt.close(fig)
        print(
            "-> Saved individual + combined coverage maps to 'coverage_maps_comparison.png'"
        )
    except ImportError:
        print("-> (Install matplotlib to generate 'coverage_maps_comparison.png')")

    # Render onboard camera views from Drone 0
    renderer = mujoco.Renderer(model, height=320, width=420)
    if cam_spec.sensor_type == "mono":
        renderer.update_scene(data, camera="cam_mono_0")
        rgb = renderer.render()
        iio.imwrite("drone0_mono_camera.png", rgb)
        print("-> Saved Drone 0 monocular camera frame to 'drone0_mono_camera.png'")
    else:
        renderer.update_scene(data, camera="cam_left_0")
        left_rgb = renderer.render()
        renderer.update_scene(data, camera="cam_right_0")
        right_rgb = renderer.render()
        stereo_pair = np.concatenate([left_rgb, right_rgb], axis=1)
        iio.imwrite("drone0_stereo_pair.png", stereo_pair)
        print(
            "-> Saved Drone 0 left+right stereo camera pair to 'drone0_stereo_pair.png'"
        )
    renderer.close()


# =====================================================================
# 5. MAIN SIMULATION LOOP
# =====================================================================


def run_simulation(
    algo_name: str,
    mode: str,
    sensor_type: str,
    scenario: str,
    custom_mesh: Optional[str],
    num_drones: int,
    max_steps: int,
):
    cam_spec = CameraSpec(sensor_type=sensor_type)
    algo_fn = ALGORITHMS[algo_name]

    xml_str = build_swarm_xml(
        num_drones, cam_spec, scenario=scenario, custom_mesh_path=custom_mesh
    )
    model = mujoco.MjModel.from_xml_string(xml_str)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    grid = CoverageGrid(num_drones=num_drones, size=16.0, resolution=45)
    grid.mark_obstacles(model, data)

    drones = [
        DroneState(id=i, pos=np.zeros(3), vel=np.zeros(3)) for i in range(num_drones)
    ]

    print(
        f"=== Swarm Sim | Mode: {mode.upper()} | Algo: {algo_name.upper()} | "
        f"Sensor: {sensor_type.upper()} | Scenario: {scenario.upper()} ==="
    )

    def step_simulation(step_idx: int):
        for i, d in enumerate(drones):
            qpos_adr, qvel_adr = i * 7, i * 6
            d.pos = data.qpos[qpos_adr : qpos_adr + 3].copy()
            d.vel = data.qvel[qvel_adr : qvel_adr + 3].copy()
            data.qpos[qpos_adr + 3 : qpos_adr + 7] = [1.0, 0.0, 0.0, 0.0]
            data.qvel[qvel_adr + 3 : qvel_adr + 6] = [0.0, 0.0, 0.0]

        # Simulate Drone 0 failure at step 500 to test self-healing redistribution
        if step_idx == 500 and len(drones) > 1:
            drones[0].active = False
            geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "core_0")
            model.geom_rgba[geom_id] = [0.8, 0.1, 0.1, 0.4]
            print(
                f"\n[EVENT @ Step {step_idx}] Drone 0 FAILED! Neighboring drones adapting coverage..."
            )

        # Update camera coverage + P2P map sharing every 4 steps
        if step_idx % 4 == 0:
            grid.update_from_cameras(drones, cam_spec, model, data)

        # Replan targets every 10 steps (0.1s)
        if step_idx % 10 == 0:
            for d in drones:
                if not d.active:
                    d.target = np.array([d.pos[0], d.pos[1], 0.15], dtype=np.float64)
                    continue

                local_neighbors = [
                    nb
                    for nb in drones
                    if nb.id != d.id and np.linalg.norm(nb.pos - d.pos) <= d.comm_radius
                ]
                new_target = algo_fn(d, local_neighbors, grid)
                if new_target is not None:
                    d.target = np.asarray(new_target, dtype=np.float64)
                else:
                    d.target = np.array([d.pos[0], d.pos[1], 2.4], dtype=np.float64)

                data.mocap_pos[d.id] = [d.target[0], d.target[1], 0.05]

        # PD Position Controller
        kp, kd, mass = 6.0, 4.5, 1.2
        for i, d in enumerate(drones):
            vel_cmd = np.clip((d.target - d.pos) * kp, -7.0, 7.0)
            acc_cmd = vel_cmd - kd * d.vel
            data.ctrl[i * 3 : i * 3 + 3] = [
                mass * acc_cmd[0],
                mass * acc_cmd[1],
                mass * (acc_cmd[2] + 9.81),
            ]

        mujoco.mj_step(model, data)

        if step_idx % 250 == 0:
            indiv_str = " | ".join(
                [
                    f"D{i}:{grid.get_drone_coverage_pct(i):.0f}%"
                    for i in range(num_drones)
                ]
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
    parser.add_argument(
        "--algo", type=str, default="voronoi", choices=list(ALGORITHMS.keys())
    )
    parser.add_argument(
        "--mode", type=str, default="headless", choices=["headless", "visual"]
    )
    parser.add_argument(
        "--sensor", type=str, default="stereo", choices=["mono", "stereo"]
    )
    parser.add_argument(
        "--scenario",
        type=str,
        default="urban",
        choices=["flat", "urban", "ruins", "custom_mesh"],
    )
    parser.add_argument(
        "--mesh-file",
        type=str,
        default=None,
        help="Path to custom .stl or .obj mesh file",
    )
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

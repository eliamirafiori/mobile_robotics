"""
Decentralized 3D Drone Swarm Area Coverage Simulation (MuJoCo)
==============================================================
Simulates a team of quadrotor drones equipped with downward-facing monocular
or stereo cameras exploring a 3D environment with obstacles.

New Capabilities in v3:
  1. Live Area Assignment Visualization (`--show-assignment` in visual mode):
     Draws color-coded territory tiles directly on the MuJoCo 3D floor via
     `viewer.user_scn` at every step so you can watch Voronoi cells / dynamic
     assignments shift in real time when a drone fails or moves.
  2. Multi-Drone SLAM & 3D Reconstruction Data Extractor (`--save-slam-data`):
     Extracts synchronized RGB frames, metric depth maps (meters), camera
     intrinsics (K matrix), and 6-DoF SE(3) camera poses (OpenCV/TUM convention)
     for every drone at configurable intervals.
"""

import argparse
import json
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

    sensor_type: str = (
        "stereo"  # "mono" (single lens) or "stereo" (overlapping L/R lenses)
    )
    fovy: float = 65.0  # Vertical field of view (degrees)
    width: int = 320  # Image width in pixels
    height: int = 240  # Image height in pixels
    baseline: float = 0.12  # Physical distance between left & right stereo lenses (m)
    max_depth: float = 5.5  # Maximum effective sensing range for coverage (m)
    min_depth: float = 0.4  # Minimum blind-spot distance near the lens (m)

    @property
    def aspect(self) -> float:
        return self.width / float(self.height)


@dataclass
class DroneState:
    """Stores the real-time physical and decentralized networking state of one drone."""

    id: int
    pos: np.ndarray  # Current 3D world position [x, y, z] (m)
    vel: np.ndarray  # Current 3D linear velocity [vx, vy, vz] (m/s)
    yaw: float = 0.0  # Heading angle around Z-axis (rad)
    active: bool = (
        True  # False if battery depleted / crashed (triggers swarm self-healing)
    )
    comm_radius: float = 9.0  # Peer-to-peer radio communication range (m)
    target: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))


@dataclass
class CoverageGrid:
    """
    2D discretized ground map tracking:
      1. Individual coverage (`drone_sole_visited`): cells seen by each drone's camera.
      2. Decentralized belief (`drone_belief_uncertainty`): local map of unexplored cells.
      3. Live area assignment (`cell_owner`): which drone currently owns each grid cell
         (-1 = obstacle or unassigned, 0..N-1 = assigned drone ID).
    """

    num_drones: int
    size: float = 16.0  # Square arena side length in meters (-size/2 to +size/2)
    resolution: int = 40  # Grid cells per axis (40x40 = 1,600 cells)

    def __post_init__(self):
        half = self.size / 2.0
        self.cell_size = self.size / self.resolution
        step = self.cell_size * 0.5
        coords = np.linspace(-half + step, half - step, self.resolution)
        gx, gy = np.meshgrid(coords, coords)

        # (N_cells, 2) array of [x, y] coordinates for every ground cell center
        self.cell_centers = np.stack([gx.ravel(), gy.ravel()], axis=-1).astype(
            np.float64
        )
        self.num_cells = len(self.cell_centers)

        # Precompute 3D ground target points (z = 0.02m) once to speed up raycasting
        self.ground_points_3d = np.column_stack(
            [self.cell_centers, np.full(self.num_cells, 0.02, dtype=np.float64)]
        )

        # Boolean mask of cells physically visited by each individual drone
        self.drone_sole_visited = np.zeros(
            (self.num_drones, self.num_cells), dtype=bool
        )

        # Uncertainty map per drone: 1.0 = completely unexplored, 0.05 = thoroughly scanned
        self.drone_belief_uncertainty = np.ones(
            (self.num_drones, self.num_cells), dtype=np.float32
        )

        # Current owner drone ID for each cell (-1 = obstacle/unassigned)
        self.cell_owner = np.full(self.num_cells, -1, dtype=np.int32)

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
            hit_dist = mujoco.mj_ray(
                model, data, ray_origin, ray_dir, None, 1, -1, geom_id_out
            )
            if 0.0 < hit_dist < 9.8:
                self.free_cells[idx] = False
                self.drone_belief_uncertainty[:, idx] = 0.0

        self.obstacle_centers = self.cell_centers[~self.free_cells]
        self.total_free_cells = max(1, int(np.sum(self.free_cells)))

    def update_area_assignments(self, drones: List[DroneState], algo_name: str) -> None:
        """
        Computes which drone currently 'owns' each ground cell for live visualization
        and workload monitoring:
          - For `voronoi`: exact nearest-active-drone partition.
          - For `frontier` / `spiral_inward`: nearest active drone target / sector.
        """
        self.cell_owner.fill(-1)
        active_drones = [d for d in drones if d.active]
        if not active_drones:
            return

        free_indices = np.where(self.free_cells)[0]
        free_pts = self.cell_centers[free_indices]

        if algo_name == "voronoi":
            # Partition by active drone positions (Voronoi diagram)
            ref_points = np.array([d.pos[:2] for d in active_drones])
        else:
            # Partition by active drones' current assigned target waypoints
            ref_points = np.array([d.target[:2] for d in active_drones])

        # Distance from every free cell to every active drone reference point: (N_free, N_active)
        dists = np.linalg.norm(free_pts[:, None, :] - ref_points[None, :, :], axis=2)
        nearest_active_idx = np.argmin(dists, axis=1)
        active_ids = np.array([d.id for d in active_drones], dtype=np.int32)
        self.cell_owner[free_indices] = active_ids[nearest_active_idx]

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

        for d in drones:
            if not d.active:
                continue

            rel_xy = self.cell_centers - d.pos[:2]
            height = max(float(d.pos[2]), 0.2)
            dists_3d = np.sqrt(np.einsum("ij,ij->i", rel_xy, rel_xy) + height * height)

            cos_y, sin_y = np.cos(-d.yaw), np.sin(-d.yaw)
            local_x = cos_y * rel_xy[:, 0] - sin_y * rel_xy[:, 1]
            local_y = sin_y * rel_xy[:, 0] + cos_y * rel_xy[:, 1]

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

            cam_origin = np.array(
                [d.pos[0], d.pos[1], d.pos[2] - 0.08], dtype=np.float64
            )
            ray_vecs = self.ground_points_3d[candidate_indices] - cam_origin
            ray_lengths = np.linalg.norm(ray_vecs, axis=1)
            ray_dirs = ray_vecs / ray_lengths[:, None]

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
                    hd = mujoco.mj_ray(
                        model,
                        data,
                        cam_origin,
                        ray_dirs[k],
                        None,
                        1,
                        body_id,
                        geom_id_out,
                    )
                    unoccluded[k] = (hd < 0.0) or (hd >= (ray_lengths[k] - 0.15))

            visible_cells = candidate_indices[unoccluded]
            if len(visible_cells) > 0:
                self.drone_sole_visited[d.id, visible_cells] = True
                self.drone_belief_uncertainty[d.id, visible_cells] = np.maximum(
                    0.05,
                    self.drone_belief_uncertainty[d.id, visible_cells]
                    * (1.0 - decay_rate),
                )

        # Decentralized Peer-to-Peer Map Merging
        active_ids = [d.id for d in drones if d.active]
        if len(active_ids) > 1:
            positions = np.array([drones[i].pos for i in active_ids])
            for idx_a in range(len(active_ids)):
                i = active_ids[idx_a]
                dists = np.linalg.norm(
                    positions[idx_a + 1 :] - positions[idx_a], axis=1
                )
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
        covered = np.sum(self.drone_sole_visited[drone_id] & self.free_cells)
        return float(covered) / self.total_free_cells * 100.0

    @property
    def combined_visited_map(self) -> np.ndarray:
        return np.any(self.drone_sole_visited, axis=0)

    @property
    def combined_coverage_pct(self) -> float:
        covered = np.sum(self.combined_visited_map & self.free_cells)
        return float(covered) / self.total_free_cells * 100.0


# =====================================================================
# 2. SWAPPABLE COVERAGE ALGORITHMS (PLUGIN REGISTRY)
# =====================================================================

AlgorithmFunc = Callable[[DroneState, List[DroneState], CoverageGrid], np.ndarray]
ALGORITHMS: Dict[str, AlgorithmFunc] = {}


def register_algorithm(name: str):
    """
    Decorator that registers a coverage planning function in the global
    ALGORITHMS registry so it can be selected via `--algo <name>`.
    """

    def decorator(func: AlgorithmFunc) -> AlgorithmFunc:
        ALGORITHMS[name] = func
        return func

    return decorator


def apply_obstacle_clearance(
    raw_target_xy: np.ndarray, grid: CoverageGrid, min_clearance_m: float = 1.3
) -> np.ndarray:
    """
    Pushes a candidate 2D waypoint away from nearby 3D building footprints and
    clamps the final coordinate safely inside the outer arena boundary walls.
    """
    obstacle_positions = grid.obstacle_centers
    if len(obstacle_positions) > 0:
        vectors_from_obstacles = raw_target_xy - obstacle_positions
        distances_to_obstacles = np.linalg.norm(vectors_from_obstacles, axis=1)
        within_clearance_mask = distances_to_obstacles < min_clearance_m

        if np.any(within_clearance_mask):
            # Weight repulsion inversely by distance so closer obstacles push harder
            repulsion_weights = (
                min_clearance_m - distances_to_obstacles[within_clearance_mask]
            ) / (distances_to_obstacles[within_clearance_mask] + 1e-3)
            total_repulsion_vector = np.sum(
                vectors_from_obstacles[within_clearance_mask]
                * repulsion_weights[:, None],
                axis=0,
            )
            repulsion_magnitude = np.linalg.norm(total_repulsion_vector)
            if repulsion_magnitude > 1e-4:
                repulsion_direction = total_repulsion_vector / repulsion_magnitude
                raw_target_xy = raw_target_xy + repulsion_direction * 0.65

    arena_half_limit = (grid.size * 0.5) - 0.8
    return np.clip(raw_target_xy, -arena_half_limit, arena_half_limit)


@register_algorithm("voronoi")
def voronoi_coverage(
    ego: DroneState, neighbors: List[DroneState], grid: CoverageGrid
) -> np.ndarray:
    """
    Decentralized Dynamic Voronoi Partitioning + Hybrid Centroid/Sweep:
      1. Territory Partitioning: Claims all traversable cells closer to `ego`
         than to any active neighbor within communication range. If a neighbor
         fails (`nb.active == False`), its cells are absorbed automatically.
      2. Hybrid Waypoint Selection: Blends the uncertainty-weighted Voronoi
         centroid (30%) with the highest-scoring unvisited cell inside the
         partition (70%) so the drone sweeps actively rather than hovering.
    """
    all_cell_positions = grid.cell_centers
    ego_uncertainty_map = grid.drone_belief_uncertainty[ego.id]

    # 1. Partition the map into ego's Voronoi cell against all active neighbors
    distance_from_ego_to_cells = np.linalg.norm(
        all_cell_positions - ego.pos[:2], axis=1
    )
    belongs_to_ego_voronoi_mask = grid.free_cells.copy()

    for neighbor in neighbors:
        if neighbor.active and neighbor.id != ego.id:
            distance_from_neighbor_to_cells = np.linalg.norm(
                all_cell_positions - neighbor.pos[:2], axis=1
            )
            belongs_to_ego_voronoi_mask &= (
                distance_from_ego_to_cells <= distance_from_neighbor_to_cells
            )

    # Fallback to all free cells if ego's partition becomes momentarily empty
    if not np.any(belongs_to_ego_voronoi_mask):
        belongs_to_ego_voronoi_mask = grid.free_cells.copy()

    voronoi_cell_positions = all_cell_positions[belongs_to_ego_voronoi_mask]
    voronoi_cell_uncertainties = ego_uncertainty_map[belongs_to_ego_voronoi_mask]
    voronoi_cell_distances = distance_from_ego_to_cells[belongs_to_ego_voronoi_mask]

    # 2. Compute uncertainty-weighted Voronoi centroid (Lloyd's relaxation step)
    uncertainty_weights = (voronoi_cell_uncertainties**3) + 1e-5
    weighted_voronoi_centroid = np.average(
        voronoi_cell_positions, axis=0, weights=uncertainty_weights
    )

    # 3. Select highest-priority local sweep cell with hysteresis to prevent jitter
    distance_to_previous_target = np.linalg.norm(
        voronoi_cell_positions - ego.target[:2], axis=1
    )
    target_persistence_bonus = 0.35 * np.exp(-distance_to_previous_target / 1.5)
    local_sweep_scores = (voronoi_cell_uncertainties**2) / (
        0.35 * voronoi_cell_distances + 1.0
    ) + target_persistence_bonus
    best_local_sweep_position = voronoi_cell_positions[
        int(np.argmax(local_sweep_scores))
    ]

    # 4. Blend global centroid pull with local sweep target and enforce obstacle clearance
    blended_target_xy = (0.10 * weighted_voronoi_centroid) + (
        0.90 * best_local_sweep_position
    )
    safe_target_xy = apply_obstacle_clearance(blended_target_xy, grid)

    return np.array(
        [float(safe_target_xy[0]), float(safe_target_xy[1]), 2.4], dtype=np.float64
    )


@register_algorithm("frontier")
def frontier_coverage(
    ego: DroneState, neighbors: List[DroneState], grid: CoverageGrid
) -> np.ndarray:
    """
    Decentralized Cooperative Information-Gain Frontier Exploration:
      1. Candidate Selection: Collects unvisited cells (uncertainty > 0.25) and
         subsamples the top 80 most uncertain candidates for fast evaluation.
      2. Information Gain: Sums expected uncertainty reduction within a 2.0m
         camera footprint around each candidate.
      3. Peer Deconfliction: Penalizes candidates closer to an active neighbor or
         within the Gaussian repulsion zone of a neighbor's declared target.
    """
    all_cell_positions = grid.cell_centers
    ego_uncertainty_map = grid.drone_belief_uncertainty[ego.id]
    traversable_cell_mask = grid.free_cells

    # 1. Identify unexplored frontier cells (fallback to all free cells once explored)
    unexplored_cell_mask = (ego_uncertainty_map > 0.25) & traversable_cell_mask
    if not np.any(unexplored_cell_mask):
        unexplored_cell_mask = traversable_cell_mask.copy()

    candidate_cell_indices = np.where(unexplored_cell_mask)[0]
    if len(candidate_cell_indices) > 80:
        highest_uncertainty_order = np.argsort(
            ego_uncertainty_map[candidate_cell_indices]
        )[-80:]
        candidate_cell_indices = candidate_cell_indices[highest_uncertainty_order]

    candidate_positions = all_cell_positions[candidate_cell_indices]
    traversable_positions = all_cell_positions[traversable_cell_mask]
    traversable_uncertainties = ego_uncertainty_map[traversable_cell_mask]

    # 2. Evaluate total expected information gain within a 2.0m radius of each candidate
    candidate_to_traversable_distances = np.linalg.norm(
        candidate_positions[:, None, :] - traversable_positions[None, :, :], axis=2
    )
    expected_information_gain = np.sum(
        (candidate_to_traversable_distances <= 2.0)
        * traversable_uncertainties[None, :],
        axis=1,
    )

    # 3. Compute travel distance cost and momentum toward ego's current waypoint
    distance_from_ego_to_candidates = np.linalg.norm(
        candidate_positions - ego.pos[:2], axis=1
    )
    distance_to_previous_target = np.linalg.norm(
        candidate_positions - ego.target[:2], axis=1
    )
    target_persistence_bonus = 2.0 * np.exp(-distance_to_previous_target / 1.5)

    # 4. Penalize candidates claimed by or closer to active neighboring drones
    neighbor_deconfliction_penalty = np.zeros(
        len(candidate_positions), dtype=np.float64
    )
    for neighbor in neighbors:
        if not neighbor.active or neighbor.id == ego.id:
            continue

        distance_from_neighbor_to_candidates = np.linalg.norm(
            candidate_positions - neighbor.pos[:2], axis=1
        )
        distance_from_neighbor_target_to_candidates = np.linalg.norm(
            candidate_positions - neighbor.target[:2], axis=1
        )

        # Discourage chasing cells that a neighbor can reach faster
        neighbor_deconfliction_penalty += np.where(
            distance_from_neighbor_to_candidates < distance_from_ego_to_candidates,
            6.0,
            0.0,
        )
        # Repel from the neighborhood around each neighbor's active target
        neighbor_deconfliction_penalty += 12.0 * np.exp(
            -(distance_from_neighbor_target_to_candidates**2) / 5.0
        )

    # 5. Select candidate maximizing net frontier utility
    frontier_utility_scores = (
        (1.5 * expected_information_gain)
        - (0.85 * distance_from_ego_to_candidates)
        + target_persistence_bonus
        - neighbor_deconfliction_penalty
    )
    highest_utility_position = candidate_positions[
        int(np.argmax(frontier_utility_scores))
    ]
    safe_target_xy = apply_obstacle_clearance(highest_utility_position, grid)

    return np.array(
        [float(safe_target_xy[0]), float(safe_target_xy[1]), 2.4], dtype=np.float64
    )


@register_algorithm("spiral_inward")
def spiral_inward_coverage(
    ego: DroneState, neighbors: List[DroneState], grid: CoverageGrid
) -> np.ndarray:
    """
    Decentralized Outside-to-Inside Concentric Ring Sweep:
      1. Self-Healing Angular Spacing: Computes ego's phase angle around the
         circle from the sorted IDs of active peers so dropouts close automatically.
      2. Contraction Radius: Tracks the 90th-percentile radius of remaining
         unexplored cells, spiraling inward from the perimeter toward the origin.
    """
    all_cell_positions = grid.cell_centers
    ego_uncertainty_map = grid.drone_belief_uncertainty[ego.id]
    traversable_cell_mask = grid.free_cells

    # 1. Determine active swarm count and ego's angular slot in the ring
    sorted_active_drone_ids = sorted(
        [ego.id] + [neighbor.id for neighbor in neighbors if neighbor.active]
    )
    active_swarm_count = len(sorted_active_drone_ids)
    ego_ring_slot_index = sorted_active_drone_ids.index(ego.id)

    # 2. Determine target spiral radius from the outermost unexplored cells
    cell_radii_from_center = np.linalg.norm(all_cell_positions, axis=1)
    unexplored_cell_mask = (ego_uncertainty_map > 0.25) & traversable_cell_mask
    if np.any(unexplored_cell_mask):
        target_ring_radius = float(
            np.percentile(cell_radii_from_center[unexplored_cell_mask], 90)
        )
    else:
        target_ring_radius = grid.size * 0.25

    # 3. Advance angular sweep proportionally to global exploration progress
    average_arena_uncertainty = float(
        np.mean(ego_uncertainty_map[traversable_cell_mask])
    )
    cumulative_sweep_angle = (1.0 - average_arena_uncertainty) * (6.0 * np.pi)
    desired_angular_position = cumulative_sweep_angle + (
        2.0 * np.pi * ego_ring_slot_index / active_swarm_count
    )

    ideal_spiral_waypoint_xy = np.array(
        [
            target_ring_radius * np.cos(desired_angular_position),
            target_ring_radius * np.sin(desired_angular_position),
        ],
        dtype=np.float64,
    )

    # 4. Snap to the highest-uncertainty free cell near the ideal spiral coordinate
    traversable_positions = all_cell_positions[traversable_cell_mask]
    traversable_uncertainties = ego_uncertainty_map[traversable_cell_mask]
    distance_to_ideal_waypoint = np.linalg.norm(
        traversable_positions - ideal_spiral_waypoint_xy, axis=1
    )

    spiral_cell_scores = (traversable_uncertainties * 2.5) - (
        0.6 * distance_to_ideal_waypoint
    )
    best_spiral_cell_xy = traversable_positions[int(np.argmax(spiral_cell_scores))]
    safe_target_xy = apply_obstacle_clearance(best_spiral_cell_xy, grid)

    return np.array(
        [float(safe_target_xy[0]), float(safe_target_xy[1]), 2.4], dtype=np.float64
    )


@register_algorithm("nearest_unvisited")
def nearest_unvisited_coverage(
    ego: DroneState, neighbors: List[DroneState], grid: CoverageGrid
) -> np.ndarray:
    """
    Baseline 1: Greedy Nearest Unvisited Cell
    Ignores neighbors completely and simply flies to the closest unvisited cell.
    """
    all_cell_positions = grid.cell_centers
    ego_uncertainty_map = grid.drone_belief_uncertainty[ego.id]

    # Find all free cells that have not been visited yet
    unvisited_mask = (ego_uncertainty_map > 0.25) & grid.free_cells

    # If everything is already visited, just hover in place
    if not np.any(unvisited_mask):
        return np.array([float(ego.pos[0]), float(ego.pos[1]), 2.4], dtype=np.float64)

    unvisited_positions = all_cell_positions[unvisited_mask]
    distances_to_ego = np.linalg.norm(unvisited_positions - ego.pos[:2], axis=1)

    # Pick the closest unvisited cell
    closest_cell_xy = unvisited_positions[int(np.argmin(distances_to_ego))]
    safe_target_xy = apply_obstacle_clearance(closest_cell_xy, grid)

    return np.array(
        [float(safe_target_xy[0]), float(safe_target_xy[1]), 2.4], dtype=np.float64
    )


@register_algorithm("lawnmower")
def lawnmower_coverage(
    ego: DroneState, neighbors: List[DroneState], grid: CoverageGrid
) -> np.ndarray:
    """
    Baseline 2: Strip-Based Lawnmower (Boustrophedon) Sweep
    Divides the Y-axis into horizontal bands among active drones and sweeps
    left-to-right on even rows, right-to-left on odd rows.
    """
    all_cell_positions = grid.cell_centers
    ego_uncertainty_map = grid.drone_belief_uncertainty[ego.id]

    # 1. Assign each active drone a horizontal band along the Y-axis
    active_ids = sorted([ego.id] + [nb.id for nb in neighbors if nb.active])
    my_band_index = active_ids.index(ego.id)
    num_bands = len(active_ids)

    half_size = grid.size * 0.5
    band_height = grid.size / num_bands
    y_min = -half_size + (my_band_index * band_height)
    y_max = y_min + band_height

    # 2. Filter unvisited cells inside ego's horizontal band (fallback to whole map if band is done)
    in_band_mask = (
        (all_cell_positions[:, 1] >= y_min)
        & (all_cell_positions[:, 1] < y_max)
        & (ego_uncertainty_map > 0.25)
        & grid.free_cells
    )
    if not np.any(in_band_mask):
        in_band_mask = (ego_uncertainty_map > 0.25) & grid.free_cells
    if not np.any(in_band_mask):
        return np.array([float(ego.pos[0]), float(ego.pos[1]), 2.4], dtype=np.float64)

    cand_pts = all_cell_positions[in_band_mask]

    # 3. Sort cells into a snake/lawnmower order:
    # Group by Y row (rounded to 0.5m), then alternate X direction every row
    row_indices = np.round((cand_pts[:, 1] + half_size) / 0.8).astype(int)
    min_row = int(np.min(row_indices))
    row_pts = cand_pts[row_indices == min_row]

    # Even rows sweep left-to-right (min X); odd rows sweep right-to-left (max X)
    if min_row % 2 == 0:
        next_cell_xy = row_pts[int(np.argmin(row_pts[:, 0]))]
    else:
        next_cell_xy = row_pts[int(np.argmax(row_pts[:, 0]))]

    safe_target_xy = apply_obstacle_clearance(next_cell_xy, grid)
    return np.array(
        [float(safe_target_xy[0]), float(safe_target_xy[1]), 2.4], dtype=np.float64
    )


# =====================================================================
# 3. LIVE VIEWER OVERLAY: RENDER ASSIGNED TERRITORY ON MUJOCO FLOOR
# =====================================================================

DRONE_PALETTE_RGBA = [
    [0.20, 0.85, 0.35, 0.30],  # Drone 0: Green
    [0.20, 0.65, 1.00, 0.30],  # Drone 1: Blue
    [1.00, 0.60, 0.15, 0.30],  # Drone 2: Orange
    [0.95, 0.25, 0.80, 0.30],  # Drone 3: Magenta
    [0.15, 0.90, 0.90, 0.30],  # Drone 4: Cyan
    [0.95, 0.90, 0.20, 0.30],  # Drone 5: Yellow
]


def render_live_assignment_overlay(viewer_handle, grid: CoverageGrid) -> None:
    """
    Draws translucent color-coded ground tiles in `viewer.user_scn` showing which
    drone currently owns each cell of the arena at the current time step.
    Brighter alpha indicates cells already visited; dimmer alpha indicates assigned
    but still unvisited territory.
    """
    scn = viewer_handle.user_scn
    scn.ngeom = 0  # Reset custom debug geoms each frame

    half_tile = grid.cell_size * 0.46
    tile_size = np.array([half_tile, half_tile, 0.004], dtype=np.float64)
    identity_mat = np.eye(3, dtype=np.float64).ravel()

    combined_visited = grid.combined_visited_map
    max_geoms = scn.maxgeom

    for idx in range(grid.num_cells):
        if scn.ngeom >= max_geoms:
            break
        owner = grid.cell_owner[idx]
        if owner < 0:
            continue

        rgba = list(DRONE_PALETTE_RGBA[owner % len(DRONE_PALETTE_RGBA)])
        # Make visited cells more vivid than unvisited assigned cells
        rgba[3] = 0.55 if combined_visited[idx] else 0.18

        pos = np.array(
            [grid.cell_centers[idx, 0], grid.cell_centers[idx, 1], 0.006],
            dtype=np.float64,
        )
        mujoco.mjv_initGeom(
            scn.geoms[scn.ngeom],
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=tile_size,
            pos=pos,
            mat=identity_mat,
            rgba=np.array(rgba, dtype=np.float32),
        )
        scn.ngeom += 1


# =====================================================================
# 4. SLAM / 3D RECONSTRUCTION SENSOR EXTRACTOR
# =====================================================================


class MultiDroneSlamExtractor:
    """
    Extracts synchronized RGB images, metric depth maps, camera intrinsics (K),
    and 6-DoF SE(3) camera poses (in standard OpenCV/SLAM coordinates) for each drone.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        cam_spec: CameraSpec,
        num_drones: int,
        output_dir: str = "slam_dataset",
    ):
        self.model = model
        self.cam_spec = cam_spec
        self.num_drones = num_drones
        self.output_dir = output_dir

        # Create separate renderers for RGB and metric Depth
        self.rgb_renderer = mujoco.Renderer(
            model, height=cam_spec.height, width=cam_spec.width
        )
        self.depth_renderer = mujoco.Renderer(
            model, height=cam_spec.height, width=cam_spec.width
        )
        self.depth_renderer.enable_depth_rendering()

        # Compute camera intrinsic matrix K from vertical FOV and resolution
        fy = (cam_spec.height * 0.5) / np.tan(np.deg2rad(cam_spec.fovy * 0.5))
        fx = fy  # Square pixels in MuJoCo
        cx = cam_spec.width * 0.5
        cy = cam_spec.height * 0.5
        self.K = np.array(
            [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64
        )

        # MuJoCo camera frame (+X right, +Y up, -Z forward) -> OpenCV/SLAM (+X right, +Y down, +Z forward)
        self._mj_to_cv = np.diag([1.0, -1.0, -1.0])

        os.makedirs(output_dir, exist_ok=True)
        for i in range(num_drones):
            drone_dir = os.path.join(output_dir, f"drone_{i}")
            os.makedirs(os.path.join(drone_dir, "rgb"), exist_ok=True)
            os.makedirs(os.path.join(drone_dir, "depth"), exist_ok=True)
            with open(
                os.path.join(drone_dir, "intrinsics.json"), "w", encoding="utf-8"
            ) as f:
                json.dump(
                    {
                        "width": cam_spec.width,
                        "height": cam_spec.height,
                        "fovy_deg": cam_spec.fovy,
                        "baseline_m": (
                            cam_spec.baseline
                            if cam_spec.sensor_type == "stereo"
                            else 0.0
                        ),
                        "K": self.K.tolist(),
                    },
                    f,
                    indent=2,
                )

    def capture_step(
        self, data: mujoco.MjData, step_idx: int, drones: List[DroneState]
    ) -> Dict[int, dict]:
        """
        Captures RGB, metric depth (float32 meters), and SE(3) pose for every active drone.
        Returns a dictionary keyed by `drone_id` for direct in-memory SLAM pipelines
        and writes standard files to disk (`slam_dataset/drone_i/`).
        """
        packet: Dict[int, dict] = {}
        timestamp = step_idx * self.model.opt.timestep

        for d in drones:
            if not d.active:
                continue

            cam_name = (
                f"cam_mono_{d.id}"
                if self.cam_spec.sensor_type == "mono"
                else f"cam_left_{d.id}"
            )
            cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)

            # 1. Render RGB (H, W, 3) uint8
            self.rgb_renderer.update_scene(data, camera=cam_name)
            rgb = self.rgb_renderer.render()

            # 2. Render Metric Depth (H, W) float32 in meters
            self.depth_renderer.update_scene(data, camera=cam_name)
            depth_m = self.depth_renderer.render()

            # 3. Extract 4x4 Camera-to-World Pose T_wc in OpenCV/SLAM frame
            pos_w = data.cam_xpos[cam_id].copy()
            rot_mj = data.cam_xmat[cam_id].reshape(3, 3)
            rot_cv = rot_mj @ self._mj_to_cv

            T_wc = np.eye(4, dtype=np.float64)
            T_wc[:3, :3] = rot_cv
            T_wc[:3, 3] = pos_w

            # Save RGB PNG, 16-bit millimeter Depth PNG, and append pose log
            drone_dir = os.path.join(self.output_dir, f"drone_{d.id}")
            frame_tag = f"{step_idx:06d}"
            iio.imwrite(os.path.join(drone_dir, "rgb", f"{frame_tag}.png"), rgb)

            # Standard TUM/Open3D depth encoding: uint16 millimeters (1000 = 1.0 meter)
            depth_mm = np.clip(depth_m * 1000.0, 0, 65535).astype(np.uint16)
            iio.imwrite(os.path.join(drone_dir, "depth", f"{frame_tag}.png"), depth_mm)

            with open(
                os.path.join(drone_dir, "poses_se3.txt"), "a", encoding="utf-8"
            ) as f:
                flat_T = " ".join(f"{val:.6f}" for val in T_wc.ravel())
                f.write(f"{timestamp:.4f} {frame_tag} {flat_T}\n")

            packet[d.id] = {
                "timestamp": timestamp,
                "rgb": rgb,
                "depth_m": depth_m,
                "K": self.K,
                "T_wc": T_wc,
            }

        return packet

    def close(self) -> None:
        self.rgb_renderer.close()
        self.depth_renderer.close()


# =====================================================================
# 5. 3D SCENARIO & MUJOCO XML SCENE BUILDER
# =====================================================================


def build_scenario_obstacles_xml(
    scenario: str, custom_mesh_path: Optional[str] = None
) -> Tuple[str, str]:
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

    if (
        scenario == "custom_mesh"
        and custom_mesh_path
        and os.path.exists(custom_mesh_path)
    ):
        assets = (
            f'<mesh name="custom_env_mesh" file="{custom_mesh_path}" scale="1 1 1"/>'
        )
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
                     rgb1="0.16 0.18 0.21" rgb2="0.20 0.22 0.26"/>
            <material name="floor_mat" texture="grid_tex" texrepeat="8 8" reflectance="0.05"/>
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
# 6. REPORTING: COVERAGE COMPARISON PLOTS & ONBOARD CAMERA SNAPSHOTS
# =====================================================================


def save_coverage_and_camera_reports(
    grid: CoverageGrid,
    drones: List[DroneState],
    cam_spec: CameraSpec,
    model: mujoco.MjModel,
    data: mujoco.MjData,
) -> None:
    res = grid.resolution
    half = grid.size * 0.5
    obstacle_mask_2d = (~grid.free_cells).reshape(res, res)

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        num_d = len(drones)
        fig, axes = plt.subplots(
            1, num_d + 1, figsize=(3.6 * (num_d + 1), 3.8), dpi=140
        )

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
            ax.set_title(
                f"Drone {i} Only\n({grid.get_drone_coverage_pct(i):.1f}% covered)"
            )
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
        print(
            "-> Saved individual + combined coverage maps to 'coverage_maps_comparison.png'"
        )
    except ImportError:
        print("-> (Install matplotlib to generate 'coverage_maps_comparison.png')")


# =====================================================================
# 7. MAIN SIMULATION LOOP & CONTROLLER
# =====================================================================


def run_simulation(
    algo_name: str,
    mode: str,
    sensor_type: str,
    scenario: str,
    custom_mesh: Optional[str],
    num_drones: int,
    max_steps: int,
    show_assignment: bool,
    save_slam_data: bool,
    slam_interval: int,
) -> None:
    cam_spec = CameraSpec(sensor_type=sensor_type)
    algo_fn = ALGORITHMS[algo_name]

    xml_str = build_swarm_xml(
        num_drones, cam_spec, scenario=scenario, custom_mesh_path=custom_mesh
    )
    model = mujoco.MjModel.from_xml_string(xml_str)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    drone_body_ids = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"drone_{i}")
        for i in range(num_drones)
    ]
    drone_core_geom_ids = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"core_{i}")
        for i in range(num_drones)
    ]

    grid = CoverageGrid(num_drones=num_drones, size=16.0, resolution=40)
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

    slam_extractor = (
        MultiDroneSlamExtractor(model, cam_spec, num_drones) if save_slam_data else None
    )

    print(
        f"=== Swarm Sim | Mode: {mode.upper()} | Algo: {algo_name.upper()} | "
        f"Sensor: {sensor_type.upper()} | Scenario: {scenario.upper()} ==="
    )

    kp, kd, mass = 6.0, 4.5, 1.2
    gravity_force = mass * 9.81

    def step_simulation(step_idx: int) -> None:
        # 1. Sync drone state from MuJoCo & stabilize attitude
        for i, d in enumerate(drones):
            qpos_adr, qvel_adr = i * 7, i * 6
            d.pos[:] = data.qpos[qpos_adr : qpos_adr + 3]
            d.vel[:] = data.qvel[qvel_adr : qvel_adr + 3]
            data.qpos[qpos_adr + 3 : qpos_adr + 7] = (1.0, 0.0, 0.0, 0.0)
            data.qvel[qvel_adr + 3 : qvel_adr + 6] = (0.0, 0.0, 0.0)

        # 2. Inject dynamic failure event at step 500
        if step_idx == 500 and len(drones) > 1:
            drones[0].active = False
            model.geom_rgba[drone_core_geom_ids[0]] = [0.8, 0.1, 0.1, 0.4]
            print(
                f"\n[EVENT @ Step {step_idx}] Drone 0 FAILED! Surviving drones absorbing territory..."
            )

        # 3. Sensor update + P2P map fusion at 25 Hz (every 4 steps)
        if step_idx % 4 == 0:
            grid.update_from_cameras(drones, drone_body_ids, cam_spec, model, data)

        # 4. Decentralized target replanning + live territory assignment at 10 Hz
        if step_idx % 10 == 0:
            all_positions = np.array([d.pos for d in drones])
            for i, d in enumerate(drones):
                if not d.active:
                    d.target[:] = (d.pos[0], d.pos[1], 0.15)
                    continue

                dists = np.linalg.norm(all_positions - d.pos, axis=1)
                local_neighbors = [
                    drones[j]
                    for j in range(num_drones)
                    if j != i and dists[j] <= d.comm_radius
                ]

                new_target = algo_fn(d, local_neighbors, grid)
                d.target[:] = (
                    new_target if new_target is not None else (d.pos[0], d.pos[1], 2.4)
                )
                data.mocap_pos[i] = (d.target[0], d.target[1], 0.05)

            grid.update_area_assignments(drones, algo_name)

        # 5. Extract RGB + Metric Depth + SE(3) Camera Pose for SLAM / 3D Reconstruction
        if slam_extractor is not None and (step_idx % slam_interval == 0):
            slam_extractor.capture_step(data, step_idx, drones)

        # 6. Low-level PD position control + gravity compensation
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
                if show_assignment:
                    render_live_assignment_overlay(v, grid)
                v.sync()
                step_idx += 1
                elapsed = time.perf_counter() - t0
                if elapsed < model.opt.timestep:
                    time.sleep(model.opt.timestep - elapsed)
    else:
        for step_idx in range(max_steps):
            step_simulation(step_idx)

    if slam_extractor is not None:
        slam_extractor.close()
        print(
            "-> Saved multi-drone RGB-D frames, intrinsics, and SE(3) poses to 'slam_dataset/'"
        )

    save_coverage_and_camera_reports(grid, drones, cam_spec, model, data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="3D Camera-Based Drone Swarm Coverage + SLAM"
    )
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
        default="flat",
        choices=["flat", "urban", "ruins", "custom_mesh"],
    )
    parser.add_argument(
        "--mesh-file", type=str, default=None, help="Path to custom .stl or .obj mesh"
    )
    parser.add_argument("--drones", type=int, default=4)
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument(
        "--show-assignment",
        action="store_true",
        default=True,
        help="Render live color-coded drone territory assignments on the floor in visual mode",
    )
    parser.add_argument(
        "--save-slam-data",
        action="store_true",
        help="Export synchronized RGB, metric depth, intrinsics, and SE(3) poses per drone",
    )
    parser.add_argument(
        "--slam-interval",
        type=int,
        default=25,
        help="Capture SLAM RGB-D frames every N physics steps (default: 25 = 4 Hz)",
    )
    args = parser.parse_args()

    run_simulation(
        algo_name=args.algo,
        mode=args.mode,
        sensor_type=args.sensor,
        scenario=args.scenario,
        custom_mesh=args.mesh_file,
        num_drones=args.drones,
        max_steps=args.steps,
        show_assignment=args.show_assignment,
        save_slam_data=args.save_slam_data,
        slam_interval=args.slam_interval,
    )

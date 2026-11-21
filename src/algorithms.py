"""
algorithms.py — Decentralized Swarm Coverage Algorithm Registry
===============================================================
Use this file purely to select existing algorithms and implement new ones.
Register any new algorithm with `@register_algorithm("name")`:

    @register_algorithm("my_algo")
    def my_algo(ego: DroneState, peers: List[DroneState], grid: CoverageGrid) -> np.ndarray:
        ...
        return np.array([x, y, 2.4])
"""

from typing import Callable, Dict, List, Optional
import numpy as np

from sim_core import CoverageGrid, DroneState, apply_obstacle_clearance

AlgoFn = Callable[[DroneState, List[DroneState], CoverageGrid], np.ndarray]
ALGORITHMS: Dict[str, AlgoFn] = {}


def register_algorithm(name: str):
    """Registers a coverage planning function under `name`."""

    def decorator(fn: AlgoFn) -> AlgoFn:
        ALGORITHMS[name] = fn
        return fn

    return decorator


def list_algorithms() -> List[str]:
    """Returns all registered algorithm names."""
    return list(ALGORITHMS.keys())


def get_algorithm(name: str) -> AlgoFn:
    """Fetches a registered algorithm function by name."""
    if name not in ALGORITHMS:
        raise KeyError(
            f"Unknown algorithm '{name}'. Available: {', '.join(list_algorithms())}"
        )
    return ALGORITHMS[name]


def select_algorithms(names: Optional[List[str]] = None) -> Dict[str, AlgoFn]:
    """Returns a subset of registered algorithms (or all coverage algorithms if `None`)."""
    if not names:
        return {k: v for k, v in ALGORITHMS.items() if k != "charge_test"}
    return {name: get_algorithm(name) for name in names}


# =====================================================================
# SHARED BATTERY HELPER (OPTIONAL FOR BATTERY-AWARE ALGORITHMS)
# =====================================================================


def manage_battery_rtb(
    ego: DroneState,
    grid: CoverageGrid,
    low_thresh: float = 30.0,
    resume_thresh: float = 95.0,
) -> Optional[np.ndarray]:
    """
    Return-To-Base (RTB) controller with landing hysteresis lock.
    Returns a 3D charging waypoint if recharging is needed, or `None` to explore.
    Automatically inactive when `grid.battery_enabled` is False or no pads exist.
    """
    if not grid.battery_enabled or len(grid.charging_stations) == 0:
        return None

    pad_dists = np.linalg.norm(grid.charging_stations - ego.pos[:2], axis=1)
    nearest_pad = grid.charging_stations[int(np.argmin(pad_dists))]
    horiz_dist = float(np.linalg.norm(nearest_pad - ego.pos[:2]))

    # Keep locked on the pad until `resume_thresh` so floor-contact jitter never aborts charging
    already_docking = (
        (horiz_dist < 1.0) and (ego.pos[2] < 1.0) and (ego.battery < resume_thresh)
    )
    needs_charge = (
        (ego.battery <= low_thresh)
        or (ego.is_charging and ego.battery < resume_thresh)
        or already_docking
    )
    if not needs_charge:
        return None

    target_z = 0.08 if horiz_dist < 0.75 else 2.4
    return np.array([float(nearest_pad[0]), float(nearest_pad[1]), target_z])


# =====================================================================
# REGISTERED COVERAGE ALGORITHMS
# =====================================================================


@register_algorithm("voronoi")
def voronoi_coverage(
    ego: DroneState, peers: List[DroneState], grid: CoverageGrid
) -> np.ndarray:
    """
    Scale-Invariant Dynamic Voronoi Partitioning + Local Sweep:
    1. Claims all traversable cells closer to `ego` than to any active peer.
    2. Caps centroid pull to <= 0.5m so drones can reach outer walls on any arena size.
    """
    pts = grid.cell_centers
    unc = grid.drone_belief_uncertainty[ego.id]

    ego_dists = np.linalg.norm(pts - ego.pos[:2], axis=1)
    my_zone = grid.free_cells.copy()
    for p in peers:
        if p.active and p.id != ego.id:
            peer_dists = np.linalg.norm(pts - p.pos[:2], axis=1)
            my_zone &= ego_dists <= peer_dists

    if not np.any(my_zone):
        my_zone = grid.free_cells.copy()

    zone_pts = pts[my_zone]
    zone_unc = unc[my_zone]
    zone_dists = ego_dists[my_zone]

    centroid = np.average(zone_pts, axis=0, weights=(zone_unc**3) + 1e-5)
    prev_dist = np.linalg.norm(zone_pts - ego.target[:2], axis=1)
    hysteresis = 0.35 * np.exp(-prev_dist / max(1.5, grid.cell_size * 2.0))
    scores = (zone_unc**2) / (0.35 * zone_dists + 1.0) + hysteresis
    sweep_pt = zone_pts[int(np.argmax(scores))]

    pull_vec = centroid - sweep_pt
    pull_norm = float(np.linalg.norm(pull_vec))
    if pull_norm > 0.5:
        pull_vec = pull_vec * (0.5 / pull_norm)

    target_xy = apply_obstacle_clearance(sweep_pt + pull_vec, grid)
    return np.array([float(target_xy[0]), float(target_xy[1]), 2.4])


@register_algorithm("frontier")
def frontier_coverage(
    ego: DroneState, peers: List[DroneState], grid: CoverageGrid
) -> np.ndarray:
    """
    Scale-Invariant Cooperative Information-Gain Frontier Exploration:
    Breaks uncertainty ties across the arena before subsampling 80 frontier candidates.
    """
    pts = grid.cell_centers
    unc = grid.drone_belief_uncertainty[ego.id]
    free = grid.free_cells

    unvisited = (unc > 0.25) & free
    if not np.any(unvisited):
        unvisited = free.copy()

    cand_idx = np.where(unvisited)[0]
    if len(cand_idx) > 80:
        rng = np.random.default_rng(seed=ego.id + int(np.sum(unvisited)))
        jittered_unc = unc[cand_idx] + rng.uniform(0.0, 1e-3, size=len(cand_idx))
        cand_idx = cand_idx[np.argsort(jittered_unc)[-80:]]

    cands = pts[cand_idx]
    free_pts = pts[free]
    free_unc = unc[free]

    dists_matrix = np.linalg.norm(cands[:, None, :] - free_pts[None, :, :], axis=2)
    info_gain = np.sum((dists_matrix <= 2.0) * free_unc[None, :], axis=1)

    ego_dists = np.linalg.norm(cands - ego.pos[:2], axis=1)
    prev_dists = np.linalg.norm(cands - ego.target[:2], axis=1)
    momentum = 2.0 * np.exp(-prev_dists / max(1.5, grid.cell_size * 2.0))

    repel_sigma_sq = max(5.0, (grid.size * 0.15) ** 2)
    penalty = np.zeros(len(cands))
    for p in peers:
        if p.active and p.id != ego.id:
            p_dists = np.linalg.norm(cands - p.pos[:2], axis=1)
            t_dists = np.linalg.norm(cands - p.target[:2], axis=1)
            penalty += np.where(p_dists < ego_dists, 6.0, 0.0)
            penalty += 12.0 * np.exp(-(t_dists**2) / repel_sigma_sq)

    utility = (1.5 * info_gain) - (0.85 * ego_dists) + momentum - penalty
    best_xy = apply_obstacle_clearance(cands[int(np.argmax(utility))], grid)
    return np.array([float(best_xy[0]), float(best_xy[1]), 2.4])


@register_algorithm("patrol")
def sector_patrol_coverage(
    ego: DroneState, peers: List[DroneState], grid: CoverageGrid
) -> np.ndarray:
    """
    Scale-Invariant Minimalist Decentralized Patrol:
    1. Locks onto current target while farther than `step_radius` and unvisited.
    2. Filters unvisited free cells beyond `step_radius`.
    3. Scores by negative distance + tiny sweep tie-breaker.
    4. Repels from active peers within 25% of arena width.
    """
    pts = grid.cell_centers
    unc = grid.drone_belief_uncertainty[ego.id]
    step_radius = max(0.8, grid.cell_size * 1.2)

    dist_to_target = float(np.linalg.norm(ego.target[:2] - ego.pos[:2]))
    nearest_cell = int(np.argmin(np.linalg.norm(pts - ego.target[:2], axis=1)))
    if (
        dist_to_target > step_radius
        and grid.free_cells[nearest_cell]
        and unc[nearest_cell] > 0.25
    ):
        return np.array([float(ego.target[0]), float(ego.target[1]), 2.4])

    ego_dists_all = np.linalg.norm(pts - ego.pos[:2], axis=1)
    unvisited = (unc > 0.25) & grid.free_cells & (ego_dists_all > step_radius)
    if not np.any(unvisited):
        unvisited = (unc > 0.25) & grid.free_cells
    if not np.any(unvisited):
        return np.array([float(ego.pos[0]), float(ego.pos[1]), 2.4])

    cands = pts[unvisited]
    ego_dists = ego_dists_all[unvisited]

    tie_breaker = 1e-2 * (cands[:, 0] + 0.1 * cands[:, 1])
    score = -ego_dists + tie_breaker

    repel_radius = max(4.0, grid.size * 0.25)
    for p in peers:
        if p.active and p.id != ego.id:
            peer_dists = np.linalg.norm(cands - p.pos[:2], axis=1)
            score -= 8.0 * np.maximum(0.0, 1.0 - (peer_dists / repel_radius))

    best_xy = apply_obstacle_clearance(cands[int(np.argmax(score))], grid)
    return np.array([float(best_xy[0]), float(best_xy[1]), 2.4])


@register_algorithm("battery_patrol")
def battery_aware_patrol(
    ego: DroneState, peers: List[DroneState], grid: CoverageGrid
) -> np.ndarray:
    """Battery-aware patrol: checks RTB charging hysteresis before running `patrol`."""
    charge_wp = manage_battery_rtb(ego, grid, low_thresh=30.0, resume_thresh=95.0)
    if charge_wp is not None:
        return charge_wp
    return sector_patrol_coverage(ego, peers, grid)


@register_algorithm("battery_voronoi")
def battery_aware_voronoi(
    ego: DroneState, peers: List[DroneState], grid: CoverageGrid
) -> np.ndarray:
    """Battery-aware Voronoi: checks RTB charging hysteresis before running `voronoi`."""
    charge_wp = manage_battery_rtb(ego, grid, low_thresh=30.0, resume_thresh=95.0)
    if charge_wp is not None:
        return charge_wp
    return voronoi_coverage(ego, peers, grid)


@register_algorithm("charge_test")
def charge_shuttle(
    ego: DroneState, _peers: List[DroneState], grid: CoverageGrid
) -> np.ndarray:
    """Diagnostic shuttle that exclusively exercises docking and recharging on pads."""
    if len(grid.charging_stations) == 0:
        return np.array([float(ego.pos[0]), float(ego.pos[1]), 2.4])

    n_pads = len(grid.charging_stations)
    pad = grid.charging_stations[ego.id % n_pads]
    offsets = np.array([[0.35, 0.0], [-0.35, 0.0], [0.0, 0.35], [0.0, -0.35]])
    offset = offsets[(ego.id // n_pads) % len(offsets)]
    landing_xy = pad + offset
    cruise_xy = pad + np.array([0.0, 2.5]) + offset

    horiz = float(np.linalg.norm(landing_xy - ego.pos[:2]))
    already_docking = (horiz < 1.0) and (ego.pos[2] < 1.0) and (ego.battery < 98.0)
    if (
        (ego.battery <= 60.0)
        or (ego.is_charging and ego.battery < 98.0)
        or already_docking
    ):
        target_z = 0.08 if horiz < 0.6 else 2.4
        return np.array([float(landing_xy[0]), float(landing_xy[1]), target_z])

    return np.array([float(cruise_xy[0]), float(cruise_xy[1]), 2.4])

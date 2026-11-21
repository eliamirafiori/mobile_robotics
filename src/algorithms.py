"""
algorithms.py — Decentralized Swarm Coverage Algorithms
=======================================================
Add new coverage algorithms here using `@register_algorithm("name")`.
Every algorithm receives one drone's local view (`ego`, `peers`, `grid`)
and returns a 3D target waypoint `[x, y, z]`.
"""

from typing import Callable, Dict, List, Optional
import numpy as np
from sim_core import CoverageGrid, DroneState, apply_obstacle_clearance

AlgoFn = Callable[[DroneState, List[DroneState], CoverageGrid], np.ndarray]
ALGORITHMS: Dict[str, AlgoFn] = {}


def register_algorithm(name: str):
    """Registers a planning function so it can be run via `--algo <name>`."""

    def decorator(fn: AlgoFn) -> AlgoFn:
        ALGORITHMS[name] = fn
        return fn

    return decorator


@register_algorithm("voronoi")
def voronoi_coverage(
    ego: DroneState, peers: List[DroneState], grid: CoverageGrid
) -> np.ndarray:
    """
    Dynamic Voronoi Partitioning + Local Sweep:
    1. Splits the map into geometric cells where each drone owns points closer
       to itself than to any active peer (auto-absorbs fallen drones' territory).
    2. Blends the uncertainty-weighted Voronoi centroid (10%) with the best
       unvisited cell inside its territory (90%) to actively sweep corners.
    """
    pts = grid.cell_centers
    unc = grid.drone_belief_uncertainty[ego.id]

    # Step 1: Claim all free cells closer to ego than to any active peer
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

    # Step 2: Weighted centroid pulls toward the densest unexplored region in our zone
    centroid = np.average(zone_pts, axis=0, weights=(zone_unc**3) + 1e-5)

    # Step 3: Greedy local sweep target with hysteresis to prevent back-and-forth jitter
    prev_dist = np.linalg.norm(zone_pts - ego.target[:2], axis=1)
    hysteresis = 0.35 * np.exp(-prev_dist / 1.5)
    scores = (zone_unc**2) / (0.35 * zone_dists + 1.0) + hysteresis
    sweep_pt = zone_pts[int(np.argmax(scores))]

    # Step 4: Combine centroid + sweep point and push safely away from buildings
    target_xy = apply_obstacle_clearance(0.10 * centroid + 0.90 * sweep_pt, grid)
    return np.array([float(target_xy[0]), float(target_xy[1]), 2.4])


@register_algorithm("frontier")
def frontier_coverage(
    ego: DroneState, peers: List[DroneState], grid: CoverageGrid
) -> np.ndarray:
    """
    Cooperative Information-Gain Frontier Exploration:
    Scores unvisited candidate cells by expected sensor gain minus travel cost
    and penalizes cells near active peers or peers' current target waypoints.
    """
    pts = grid.cell_centers
    unc = grid.drone_belief_uncertainty[ego.id]
    free = grid.free_cells

    # Step 1: Gather unvisited cells (subsample top 80 by uncertainty for speed)
    unvisited = (unc > 0.25) & free
    if not np.any(unvisited):
        unvisited = free.copy()

    cand_idx = np.where(unvisited)[0]
    if len(cand_idx) > 80:
        cand_idx = cand_idx[np.argsort(unc[cand_idx])[-80:]]

    cands = pts[cand_idx]
    free_pts = pts[free]
    free_unc = unc[free]

    # Step 2: Information gain = total uncertainty within a 2.0m camera footprint
    dists_matrix = np.linalg.norm(cands[:, None, :] - free_pts[None, :, :], axis=2)
    info_gain = np.sum((dists_matrix <= 2.0) * free_unc[None, :], axis=1)

    # Step 3: Travel cost + persistence bonus toward current target
    ego_dists = np.linalg.norm(cands - ego.pos[:2], axis=1)
    prev_dists = np.linalg.norm(cands - ego.target[:2], axis=1)
    momentum = 2.0 * np.exp(-prev_dists / 1.5)

    # Step 4: Repel from active peers' positions and target waypoints
    penalty = np.zeros(len(cands))
    for p in peers:
        if p.active and p.id != ego.id:
            p_dists = np.linalg.norm(cands - p.pos[:2], axis=1)
            t_dists = np.linalg.norm(cands - p.target[:2], axis=1)
            penalty += np.where(p_dists < ego_dists, 6.0, 0.0)
            penalty += 12.0 * np.exp(-(t_dists**2) / 5.0)

    # Step 5: Pick highest-utility frontier cell
    utility = (1.5 * info_gain) - (0.85 * ego_dists) + momentum - penalty
    best_xy = apply_obstacle_clearance(cands[int(np.argmax(utility))], grid)
    return np.array([float(best_xy[0]), float(best_xy[1]), 2.4])


@register_algorithm("patrol")
def sector_patrol_coverage(
    ego: DroneState, peers: List[DroneState], grid: CoverageGrid
) -> np.ndarray:
    """
    Minimalist 4-Step Decentralized Coverage Algorithm:
    1. Keep current target while we are still flying toward it AND it remains unvisited.
    2. Filter unvisited free cells outside our immediate 0.8m hover radius.
    3. Score remaining cells by proximity (-distance) + tiny directional tie-breaker.
    4. Repel from active peers' positions so drones split the map automatically.
    """
    # Get [x, y] coordinates of all arena grid cell centers, shape (N_cells, 2)
    pts = grid.cell_centers
    # Get this drone's local belief map [1.0 = unknown, 0.05 = well-scanned], shape (N_cells,)
    unc = grid.drone_belief_uncertainty[ego.id]

    # 1. Waypoint lock: keep flying to current target if we haven't arrived yet (> 0.8m)
    #    AND the target cell is still traversable and unvisited
    dist_to_target = float(np.linalg.norm(ego.target[:2] - ego.pos[:2]))
    # Find the array index of the cell center closest to ego.target[:2]
    # axis=1 computes the Euclidean norm across [dx, dy] for each row in pts
    nearest_cell = int(np.argmin(np.linalg.norm(pts - ego.target[:2], axis=1)))
    if (
        dist_to_target > 0.8
        and grid.free_cells[nearest_cell]
        and unc[nearest_cell] > 0.25
    ):
        return np.array([float(ego.target[0]), float(ego.target[1]), 2.4])

    # 2. Compute 2D Euclidean distance from ego's current position to every cell in the arena
    ego_dists_all = np.linalg.norm(pts - ego.pos[:2], axis=1)

    # Bitwise AND (&) combines three boolean masks:
    #   - cell is unvisited (unc > 0.25)
    #   - cell is not an obstacle (grid.free_cells)
    #   - cell is outside our immediate 0.8m hover radius (forces forward movement)
    unvisited = (unc > 0.25) & grid.free_cells & (ego_dists_all > 0.8)

    # Fallback: if no unvisited cells exist beyond 0.8m (final cleanup phase), allow closer cells
    if not np.any(unvisited):
        unvisited = (unc > 0.25) & grid.free_cells

    # If ALL free cells are visited according to ego's belief, hover in place
    if not np.any(unvisited):
        return np.array([float(ego.pos[0]), float(ego.pos[1]), 2.4])

    # Extract [x, y] coordinates and precomputed distances for only the unvisited candidate cells
    cands = pts[unvisited]
    ego_dists = ego_dists_all[unvisited]

    # 3. Primary attraction score + tie-breaker:
    #    - Closer unvisited cells get higher (less negative) scores (-ego_dists)
    #    - Tiny directional gradient (1e-2) breaks exact distance ties on the square grid
    #      by preferring +X then +Y without overriding real distance costs
    tie_breaker = 1e-2 * (cands[:, 0] + 0.1 * cands[:, 1])
    score = -ego_dists + tie_breaker

    # 4. Peer deconfliction: penalize candidates close to any active teammate so drones spread out
    for p in peers:
        if p.active and p.id != ego.id:
            # Compute 2D Euclidean distance from teammate `p` to every candidate cell
            peer_dists = np.linalg.norm(cands - p.pos[:2], axis=1)
            # Subtract up to 8.0 points if a candidate is within 4.0m of teammate `p`
            score -= np.maximum(0.0, 8.0 - 2.0 * peer_dists)

    # Pick the highest-scoring cell and push it safely away from buildings and arena walls
    best_xy = apply_obstacle_clearance(cands[int(np.argmax(score))], grid)
    # Return the new 3D waypoint at cruising altitude z = 2.4m
    return np.array([float(best_xy[0]), float(best_xy[1]), 2.4])


def manage_battery_rtb(
    ego: DroneState,
    grid: CoverageGrid,
    low_thresh: float = 30.0,
    resume_thresh: float = 95.0,
) -> Optional[np.ndarray]:
    """
    Return-To-Base (RTB) Battery Hysteresis Controller:
      - If battery <= `low_thresh`, fly to the closest charging station and descend to z = 0.20m.
      - If currently charging (`ego.is_charging`) and battery < `resume_thresh`, stay docked.
      - Returns a 3D charging waypoint if charging is needed, or `None` to resume exploration.
    """
    if len(grid.charging_stations) == 0:
        return None

    needs_charge = (ego.battery <= low_thresh) or (
        ego.is_charging and ego.battery < resume_thresh
    )
    if not needs_charge:
        return None

    # Find nearest charging station in 2D
    pad_dists = np.linalg.norm(grid.charging_stations - ego.pos[:2], axis=1)
    nearest_pad = grid.charging_stations[int(np.argmin(pad_dists))]

    # Descend to z = 0.20m once horizontally aligned so `pos[2] <= 0.45m` triggers charging
    horiz_dist = float(np.linalg.norm(nearest_pad - ego.pos[:2]))
    target_z = 0.20 if horiz_dist < 0.75 else 2.4
    return np.array([float(nearest_pad[0]), float(nearest_pad[1]), target_z])


@register_algorithm("battery_patrol")
def battery_aware_patrol(
    ego: DroneState, peers: List[DroneState], grid: CoverageGrid
) -> np.ndarray:
    """
    Battery-Aware Decentralized Patrol:
      1. Checks battery level first via `manage_battery_rtb` (RTB at 30%, resume at 95%).
      2. Otherwise runs `patrol_clean` greedy exploration with peer repulsion.
    """
    # 1. Priority override: recharge if battery is low or mid-charge cycle
    charge_wp = manage_battery_rtb(ego, grid, low_thresh=30.0, resume_thresh=95.0)
    if charge_wp is not None:
        return charge_wp

    pts = grid.cell_centers
    unc = grid.drone_belief_uncertainty[ego.id]

    # 2. Keep current target while flying toward it AND it remains unvisited
    dist_to_target = float(np.linalg.norm(ego.target[:2] - ego.pos[:2]))
    nearest_cell = int(np.argmin(np.linalg.norm(pts - ego.target[:2], axis=1)))
    if (
        dist_to_target > 0.8
        and grid.free_cells[nearest_cell]
        and unc[nearest_cell] > 0.25
    ):
        return np.array([float(ego.target[0]), float(ego.target[1]), 2.4])

    # 3. Filter unvisited free cells outside our immediate 0.8m hover radius
    ego_dists_all = np.linalg.norm(pts - ego.pos[:2], axis=1)
    unvisited = (unc > 0.25) & grid.free_cells & (ego_dists_all > 0.8)
    if not np.any(unvisited):
        unvisited = (unc > 0.25) & grid.free_cells
    if not np.any(unvisited):
        return np.array([float(ego.pos[0]), float(ego.pos[1]), 2.4])

    cands = pts[unvisited]
    ego_dists = ego_dists_all[unvisited]

    # 4. Score by proximity + directional tie-breaker + peer repulsion
    tie_breaker = 1e-2 * (cands[:, 0] + 0.1 * cands[:, 1])
    score = -ego_dists + tie_breaker

    for p in peers:
        if p.active and p.id != ego.id:
            peer_dists = np.linalg.norm(cands - p.pos[:2], axis=1)
            score -= np.maximum(0.0, 8.0 - 2.0 * peer_dists)

    best_xy = apply_obstacle_clearance(cands[int(np.argmax(score))], grid)
    return np.array([float(best_xy[0]), float(best_xy[1]), 2.4])


@register_algorithm("charge_test")
def charge_shuttle(
    ego: DroneState, _peers: List[DroneState], grid: CoverageGrid
) -> np.ndarray:
    """
    Shuttle that only exercises the charging pads.

    Each drone is pinned to one station (drone id modulo the number of pads)
    and a landing spot inside that pad, so several drones can dock without
    stacking on the same point. It cruises a few meters off the pad at
    altitude until the battery falls to 60%, then lands and stays docked
    until the battery reaches 98%.
    """
    if len(grid.charging_stations) == 0:
        return np.array([float(ego.pos[0]), float(ego.pos[1]), 2.4])

    n_pads = len(grid.charging_stations)
    pad = grid.charging_stations[ego.id % n_pads]
    # Keep shared-pad landings inside pad_radius (0.85 m).
    offsets = np.array(
        [[0.35, 0.0], [-0.35, 0.0], [0.0, 0.35], [0.0, -0.35]],
        dtype=np.float64,
    )
    offset = offsets[(ego.id // n_pads) % len(offsets)]
    landing_xy = pad + offset
    cruise_xy = pad + np.array([0.0, 2.5]) + offset

    low_thresh, resume_thresh = 60.0, 98.0
    needs_charge = (ego.battery <= low_thresh) or (
        ego.is_charging and ego.battery < resume_thresh
    )
    if needs_charge:
        horiz = float(np.linalg.norm(landing_xy - ego.pos[:2]))
        # Descend only once aligned so z <= 0.45 m can start charging.
        target_z = 0.20 if horiz < 0.6 else 2.4
        return np.array([float(landing_xy[0]), float(landing_xy[1]), target_z])

    return np.array([float(cruise_xy[0]), float(cruise_xy[1]), 2.4])

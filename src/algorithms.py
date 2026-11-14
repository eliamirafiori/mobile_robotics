"""
algorithms.py — Decentralized Swarm Coverage Algorithms
=======================================================
Add new coverage algorithms here using `@register_algorithm("name")`.
Every algorithm receives one drone's local view (`ego`, `peers`, `grid`)
and returns a 3D target waypoint `[x, y, z]`.
"""

from typing import Callable, Dict, List
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
    Simple 4-Step Decentralized Algorithm:
    1. Filter unvisited free cells from ego's belief map.
    2. Attract toward nearby cells (-distance).
    3. Repel from active peers' positions (auto-heals on peer failure).
    4. Lock onto current target direction with persistence bonus.
    """
    # Get all arena cells
    pts = grid.cell_centers
    # Get drone's uncertainty about which cells have been visited
    unc = grid.drone_belief_uncertainty[ego.id]  # [1.0 = unknown, 0.05 = well-scanned]
    # Operate bitwise AND, e.g. 1110 & 1010 -> 1010
    unvisited = (unc > 0.25) & grid.free_cells
    # If ALL cells are visited drone stays still
    if not np.any(unvisited):
        return np.array([float(ego.pos[0]), float(ego.pos[1]), 2.4])
    # Get candidates unvisited cells
    cands = pts[unvisited]

    # Keep current target if it is still unvisited and not yet reached
    nearest_cell = int(np.argmin(np.linalg.norm(pts - ego.target[:2], axis=1)))
    if grid.free_cells[nearest_cell] and unc[nearest_cell] > 0.25:
        return np.array([float(ego.target[0]), float(ego.target[1]), 2.4])

    # Computes the Euclidean distance between the ego agent and each candidate position
    # axis=1 means compute the Euclidean distance between the ego agent and each candidate position
    ego_dists = np.linalg.norm(cands - ego.pos[:2], axis=1)
    # Smooth sweep tie-breaker: prefers sweeping column-by-column (left-to-right, bottom-to-top)
    # Multiplied by 1e-2 so it NEVER overrides real distance/utility, only breaks exact ties
    tie_breaker = 1e-2 * (cands[:, 0] + 0.1 * cands[:, 1])
    score = (
        -1.0 * ego_dists + tie_breaker
    )  # Closer cells get higher (less negative) scores

    # If every drone only minimizes ego_dists, they will clump together
    # Penalize candidate cells that are close to any active peer's current position
    for p in peers:
        if p.active and p.id != ego.id:
            peer_dists = np.linalg.norm(cands - p.pos[:2], axis=1)
            # Subtract up to 8 points if a candidate is within 4m of a teammate
            score -= np.maximum(0.0, 8.0 - 2.0 * peer_dists)

    # Without persistence, two equidistant cells can cause the drone to jitter back and forth
    prev_dists = np.linalg.norm(cands - ego.target[:2], axis=1)
    # Add a small bonus near the target position
    score += 1.5 * np.exp(-prev_dists / 1.5)

    # Check if there are any obstacles
    best_xy = apply_obstacle_clearance(cands[int(np.argmax(score))], grid)
    # Return new target position for the drone
    return np.array([float(best_xy[0]), float(best_xy[1]), 2.4])


@register_algorithm("patrol2")
def sector_patrol_coverage_2(
    ego: DroneState, peers: List[DroneState], grid: CoverageGrid
) -> np.ndarray:
    """Decentralized greedy patrol with waypoint release on arrival.

    A waypoint is retained only while it is both unvisited and farther away
    than the arrival radius. Once reached, nearby/current cells are excluded
    so the drone selects a new location instead of repeatedly targeting the
    cell it is already occupying.
    """
    pts = grid.cell_centers
    unc = grid.drone_belief_uncertainty[ego.id]
    free = grid.free_cells

    unvisited = (unc > 0.25) & free
    if not np.any(unvisited):
        # Coverage is complete according to this drone's local belief.
        return np.array([float(ego.pos[0]), float(ego.pos[1]), 2.4])

    # Estimate grid spacing and use it to set a scale-aware arrival tolerance.
    # Tune the bounds if the flight controller uses a different waypoint tolerance.
    spacings = []
    for axis in (0, 1):
        coords = np.unique(np.round(pts[:, axis], decimals=6))
        steps = np.diff(coords)
        steps = steps[steps > 1e-6]
        if steps.size:
            spacings.append(float(np.min(steps)))
    arrival_radius = min(0.75, max(0.20, 0.45 * min(spacings))) if spacings else 0.35

    target_dist = float(np.linalg.norm(ego.target[:2] - ego.pos[:2]))
    target_cell = int(np.argmin(np.linalg.norm(pts - ego.target[:2], axis=1)))
    target_is_unvisited = bool(free[target_cell] and unc[target_cell] > 0.25)

    # The original code retained any unvisited target forever. It did not
    # check whether the drone had already reached it (or started on it).
    if target_is_unvisited and target_dist > arrival_radius:
        return np.array([float(ego.target[0]), float(ego.target[1]), 2.4])

    candidate_idx = np.flatnonzero(unvisited)

    # On arrival, do not immediately select the same cell again.
    if target_dist <= arrival_radius and target_is_unvisited:
        other_cells = candidate_idx != target_cell
        if np.any(other_cells):
            candidate_idx = candidate_idx[other_cells]

    cands = pts[candidate_idx]
    ego_dists = np.linalg.norm(cands - ego.pos[:2], axis=1)

    # Avoid choosing a waypoint effectively at the drone's current position.
    moving = ego_dists > arrival_radius
    if np.any(moving):
        cands = cands[moving]
        ego_dists = ego_dists[moving]
    else:
        # No unvisited cell is far enough away to be a meaningful next waypoint.
        return np.array([float(ego.pos[0]), float(ego.pos[1]), 2.4])

    # Small deterministic sweep preference; distance remains the main term.
    tie_breaker = 1e-2 * (cands[:, 0] + 0.1 * cands[:, 1])
    score = -ego_dists + tie_breaker

    # Repel from active teammates to reduce clustering.
    for p in peers:
        if p.active and p.id != ego.id:
            peer_dists = np.linalg.norm(cands - p.pos[:2], axis=1)
            score -= np.maximum(0.0, 8.0 - 2.0 * peer_dists)

    # Retain a mild preference for the previous waypoint to reduce jitter.
    prev_dists = np.linalg.norm(cands - ego.target[:2], axis=1)
    score += 1.5 * np.exp(-prev_dists / 1.5)

    best_xy = apply_obstacle_clearance(cands[int(np.argmax(score))], grid)
    return np.array([float(best_xy[0]), float(best_xy[1]), 2.4])

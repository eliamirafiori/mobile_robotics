"""
sim_core.py — MuJoCo 3D Drone Swarm Physics, Battery, Sensor & Rendering Engine
================================================================================
Encapsulates low-level physics, camera raycasting, battery drain/charging, and
performance metrics so research algorithms in `algorithms.py` stay clean.
"""

import json
import os
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import imageio.v3 as iio
import mujoco
import numpy as np

# =====================================================================
# 1. CORE DATA STRUCTURES
# =====================================================================


@dataclass
class CameraSpec:
    """Optical and depth parameters of each drone's downward camera."""

    sensor_type: str = "stereo"  # "mono" or "stereo"
    fovy: float = 65.0  # Vertical FOV (deg)
    width: int = 320  # Image width (px)
    height: int = 240  # Image height (px)
    baseline: float = 0.12  # Stereo lens separation (m)
    max_depth: float = 5.5  # Max effective sensing range (m)
    min_depth: float = 0.4  # Near blind-spot cutoff (m)

    @property
    def aspect(self) -> float:
        return self.width / float(self.height)


@dataclass
class DroneState:
    """Real-time physical, networking, and battery state of a single drone."""

    id: int
    pos: np.ndarray  # [x, y, z] position (m)
    vel: np.ndarray  # [vx, vy, vz] velocity (m/s)
    yaw: float = 0.0  # Heading around Z-axis (rad)
    active: bool = True  # False on simulated failure or empty battery
    comm_radius: float = 9.0  # P2P radio range (m)
    target: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))

    # --- Battery & Charging Telemetry ---
    battery: float = 100.0  # State of Charge: 0.0% (empty) to 100.0% (full)
    is_charging: bool = False  # True while landed on an active charging pad
    energy_used: float = 0.0  # Cumulative battery % drained during flight
    charge_time: float = 0.0  # Total simulated seconds spent recharging
    charge_visits: int = 0  # Number of docking cycles at a station
    depleted: bool = False  # True if battery hit 0.0% mid-mission


@dataclass
class CoverageGrid:
    """Discretized ground map tracking coverage, uncertainty beliefs, and stations."""

    num_drones: int
    size: float = 16.0
    resolution: int = 40

    def __post_init__(self):
        half = self.size * 0.5
        self.cell_size = self.size / self.resolution
        step = self.cell_size * 0.5
        coords = np.linspace(-half + step, half - step, self.resolution)
        gx, gy = np.meshgrid(coords, coords)

        self.cell_centers = np.stack([gx.ravel(), gy.ravel()], axis=-1).astype(
            np.float64
        )
        self.num_cells = len(self.cell_centers)
        self.ground_points_3d = np.column_stack(
            [self.cell_centers, np.full(self.num_cells, 0.02, dtype=np.float64)]
        )

        # Ground-truth audit log of cells seen by each individual drone
        self.drone_sole_visited = np.zeros(
            (self.num_drones, self.num_cells), dtype=bool
        )
        # Local belief map per drone: 1.0 = unknown, 0.05 = well-scanned
        self.drone_belief_uncertainty = np.ones(
            (self.num_drones, self.num_cells), dtype=np.float32
        )
        # Current territory owner ID (-1 = obstacle/unassigned)
        self.cell_owner = np.full(self.num_cells, -1, dtype=np.int32)
        # Traversable ground mask (False where static 3D obstacles sit)
        self.free_cells = np.ones(self.num_cells, dtype=bool)
        self.obstacle_centers = np.empty((0, 2), dtype=np.float64)

        # 2D coordinates [x, y] of charging stations (populated by scenario builder)
        self.charging_stations = np.empty((0, 2), dtype=np.float64)
        self.pad_radius: float = 0.85

    def mark_obstacles(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        """Detects static worldbody obstacles via downward rays from z = 10m."""
        origin = np.array([0.0, 0.0, 10.0], dtype=np.float64)
        direction = np.array([0.0, 0.0, -1.0], dtype=np.float64)
        geom_out = np.zeros(1, dtype=np.int32)

        for idx, (cx, cy) in enumerate(self.cell_centers):
            origin[0], origin[1] = cx, cy
            hit = mujoco.mj_ray(model, data, origin, direction, None, 1, -1, geom_out)
            hit_body = model.geom_bodyid[geom_out[0]] if geom_out[0] >= 0 else 0
            # Only treat static worldbody geometry (body 0) above floor as obstacles
            if 0.0 < hit < 9.8 and hit_body == 0:
                self.free_cells[idx] = False
                self.drone_belief_uncertainty[:, idx] = 0.0

        self.obstacle_centers = self.cell_centers[~self.free_cells]
        self.total_free_cells = max(1, int(np.sum(self.free_cells)))

    def update_area_assignments(self, drones: List[DroneState], algo_name: str) -> None:
        """Assigns each free cell to the nearest active drone (or target) for visualization."""
        self.cell_owner.fill(-1)
        active = [d for d in drones if d.active]
        if not active:
            return

        free_idx = np.where(self.free_cells)[0]
        free_pts = self.cell_centers[free_idx]
        refs = np.array(
            [d.pos[:2] if algo_name == "voronoi" else d.target[:2] for d in active]
        )

        dists = np.linalg.norm(free_pts[:, None, :] - refs[None, :, :], axis=2)
        active_ids = np.array([d.id for d in active], dtype=np.int32)
        self.cell_owner[free_idx] = active_ids[np.argmin(dists, axis=1)]

    def update_from_cameras(
        self,
        drones: List[DroneState],
        body_ids: List[int],
        cam_spec: CameraSpec,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        decay_rate: float = 0.25,
    ) -> None:
        """Updates visibility via camera frustum + MuJoCo raycasting and merges P2P maps."""
        tan_y = np.tan(np.deg2rad(cam_spec.fovy * 0.5))
        tan_x = tan_y * cam_spec.aspect
        half_base = cam_spec.baseline * 0.5
        geom_out = np.zeros(1, dtype=np.int32)

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
            in_range = (
                (np.abs(local_y) <= max_y)
                & (dists_3d <= cam_spec.max_depth)
                & (dists_3d >= cam_spec.min_depth)
                & self.free_cells
            )

            if cam_spec.sensor_type == "mono":
                in_fov = in_range & (np.abs(local_x) <= max_x)
            else:
                in_fov = (
                    in_range
                    & (np.abs(local_x + half_base) <= max_x)
                    & (np.abs(local_x - half_base) <= max_x)
                )

            cand_idx = np.where(in_fov)[0]
            if len(cand_idx) == 0:
                continue

            origin = np.array([d.pos[0], d.pos[1], d.pos[2] - 0.08], dtype=np.float64)
            vecs = self.ground_points_3d[cand_idx] - origin
            lengths = np.linalg.norm(vecs, axis=1)
            dirs = vecs / lengths[:, None]

            n_rays = len(cand_idx)
            if hasattr(mujoco, "mj_multiRay"):
                hits = np.empty(n_rays, dtype=np.float64)
                geoms = np.empty(n_rays, dtype=np.int32)
                mujoco.mj_multiRay(
                    model,
                    data,
                    origin,
                    dirs.ravel(),
                    None,
                    1,
                    body_ids[d.id],
                    geoms,
                    hits,
                    None,
                    n_rays,
                    cam_spec.max_depth + 1.0,
                )
                clear = (hits < 0.0) | (hits >= (lengths - 0.15))
            else:
                clear = np.array(
                    [
                        (
                            hd := mujoco.mj_ray(
                                model,
                                data,
                                origin,
                                dirs[k],
                                None,
                                1,
                                body_ids[d.id],
                                geom_out,
                            )
                        )
                        < 0.0
                        or hd >= (lengths[k] - 0.15)
                        for k in range(n_rays)
                    ]
                )

            visible = cand_idx[clear]
            if len(visible) > 0:
                self.drone_sole_visited[d.id, visible] = True
                self.drone_belief_uncertainty[d.id, visible] = np.maximum(
                    0.05,
                    self.drone_belief_uncertainty[d.id, visible] * (1.0 - decay_rate),
                )

        # Peer-to-peer CRDT uncertainty map merge within comm_radius
        active_ids = [d.id for d in drones if d.active]
        if len(active_ids) > 1:
            positions = np.array([drones[i].pos for i in active_ids])
            for idx_a, i in enumerate(active_ids):
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
# 2. PERFORMANCE METRICS TRACKER (COVERAGE TIME & BATTERY MANAGEMENT)
# =====================================================================


class SwarmMetricsTracker:
    """Records Coverage Time milestones and Battery Management KPIs."""

    def __init__(self, dt: float):
        self.dt = dt
        self.milestones: Dict[float, Optional[Dict[str, float]]] = {
            50.0: None,
            80.0: None,
            90.0: None,
            95.0: None,
            99.0: None,
        }

    def update(self, step: int, coverage_pct: float) -> None:
        for target_pct, recorded in self.milestones.items():
            if recorded is None and coverage_pct >= target_pct:
                self.milestones[target_pct] = {
                    "step": step,
                    "time_sec": round(step * self.dt, 2),
                }

    def summarize_and_save(
        self,
        total_steps: int,
        grid: CoverageGrid,
        drones: List[DroneState],
        algo_name: str,
        scenario: str,
        out_path: str = "metrics_report.json",
    ) -> dict:
        total_energy_used = sum(d.energy_used for d in drones)
        total_charge_time = sum(d.charge_time for d in drones)
        total_charge_visits = sum(d.charge_visits for d in drones)
        depleted_count = sum(1 for d in drones if d.depleted)
        active_count = sum(1 for d in drones if d.active)
        avg_final_battery = float(np.mean([d.battery for d in drones]))
        min_final_battery = float(np.min([d.battery for d in drones]))

        report = {
            "algorithm": algo_name,
            "scenario": scenario,
            "num_drones": len(drones),
            "coverage_time_metrics": {
                "final_coverage_pct": round(grid.combined_coverage_pct, 2),
                "total_steps_run": total_steps,
                "total_sim_time_sec": round(total_steps * self.dt, 2),
                "time_to_50_pct": self.milestones[50.0],
                "time_to_80_pct": self.milestones[80.0],
                "time_to_90_pct": self.milestones[90.0],
                "time_to_95_pct": self.milestones[95.0],
                "time_to_99_pct": self.milestones[99.0],
            },
            "battery_management_metrics": {
                "total_energy_drained_pct": round(total_energy_used, 2),
                "avg_final_battery_pct": round(avg_final_battery, 2),
                "min_final_battery_pct": round(min_final_battery, 2),
                "total_charging_dockings": total_charge_visits,
                "total_charging_time_sec": round(total_charge_time, 2),
                "battery_depletion_failures": depleted_count,
                "surviving_drones": f"{active_count}/{len(drones)}",
            },
        }

        print("\n" + "=" * 64)
        print(
            f"PERFORMANCE METRICS | Algo: {algo_name.upper()} | Scenario: {scenario.upper()}"
        )
        print("-" * 64)
        print("1. COVERAGE TIME:")
        for pct, info in self.milestones.items():
            val = (
                f"{info['time_sec']}s (step {info['step']})" if info else "Not reached"
            )
            print(f"   - Time to {pct:4.0f}% coverage : {val}")
        print(
            f"   - Final coverage        : {grid.combined_coverage_pct:.1f}% in {total_steps * self.dt:.2f}s"
        )

        print("2. BATTERY MANAGEMENT:")
        print(f"   - Total energy drained  : {total_energy_used:.1f}% (across swarm)")
        print(
            f"   - Final battery (avg/min): {avg_final_battery:.1f}% / {min_final_battery:.1f}%"
        )
        print(
            f"   - Charging dockings     : {total_charge_visits} visits ({total_charge_time:.2f}s total)"
        )
        print(
            f"   - Empty-battery crashes : {depleted_count} | Surviving: {active_count}/{len(drones)}"
        )
        print("=" * 64)

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"-> Saved metrics report to '{out_path}'")
        return report


# =====================================================================
# 3. SHARED GEOMETRY HELPER
# =====================================================================


def apply_obstacle_clearance(
    target_xy: np.ndarray, grid: CoverageGrid, clearance: float = 1.3
) -> np.ndarray:
    """Pushes a 2D waypoint away from building footprints and keeps it inside arena walls."""
    obs = grid.obstacle_centers
    if len(obs) > 0:
        vecs = target_xy - obs
        dists = np.linalg.norm(vecs, axis=1)
        close = dists < clearance
        if np.any(close):
            weights = (clearance - dists[close]) / (dists[close] + 1e-3)
            push = np.sum(vecs[close] * weights[:, None], axis=0)
            norm = np.linalg.norm(push)
            if norm > 1e-4:
                target_xy = target_xy + (push / norm) * 0.65

    limit = (grid.size * 0.5) - 0.8
    return np.clip(target_xy, -limit, limit)


# =====================================================================
# 4. VIEWER OVERLAY, SLAM EXTRACTOR, XML BUILDER & SIMULATION LOOP
# =====================================================================

DRONE_PALETTE_RGBA = [
    [0.20, 0.85, 0.35, 0.30],
    [0.20, 0.65, 1.00, 0.30],
    [1.00, 0.60, 0.15, 0.30],
    [0.95, 0.25, 0.80, 0.30],
    [0.15, 0.90, 0.90, 0.30],
    [0.95, 0.90, 0.20, 0.30],
]


def render_live_assignment_overlay(viewer_handle, grid: CoverageGrid) -> None:
    """Draws color-coded floor tiles showing live drone territory ownership."""
    scn = viewer_handle.user_scn
    scn.ngeom = 0
    half_tile = grid.cell_size * 0.46
    tile_size = np.array([half_tile, half_tile, 0.004], dtype=np.float64)
    eye = np.eye(3, dtype=np.float64).ravel()
    visited = grid.combined_visited_map

    for idx in range(min(grid.num_cells, scn.maxgeom)):
        owner = grid.cell_owner[idx]
        if owner < 0:
            continue
        rgba = list(DRONE_PALETTE_RGBA[owner % len(DRONE_PALETTE_RGBA)])
        rgba[3] = 0.55 if visited[idx] else 0.18
        pos = np.array([grid.cell_centers[idx, 0], grid.cell_centers[idx, 1], 0.006])
        mujoco.mjv_initGeom(
            scn.geoms[scn.ngeom],
            mujoco.mjtGeom.mjGEOM_BOX,
            tile_size,
            pos,
            eye,
            np.array(rgba, dtype=np.float32),
        )
        scn.ngeom += 1


class MultiDroneSlamExtractor:
    """Exports synchronized RGB, metric depth, intrinsics, and SE(3) poses per drone."""

    def __init__(
        self,
        model: mujoco.MjModel,
        cam_spec: CameraSpec,
        num_drones: int,
        out_dir: str = "slam_dataset",
    ):
        self.model = model
        self.cam_spec = cam_spec
        self.out_dir = out_dir
        self.rgb_renderer = mujoco.Renderer(
            model, height=cam_spec.height, width=cam_spec.width
        )
        self.depth_renderer = mujoco.Renderer(
            model, height=cam_spec.height, width=cam_spec.width
        )
        self.depth_renderer.enable_depth_rendering()

        fy = (cam_spec.height * 0.5) / np.tan(np.deg2rad(cam_spec.fovy * 0.5))
        self.K = np.array(
            [
                [fy, 0.0, cam_spec.width * 0.5],
                [0.0, fy, cam_spec.height * 0.5],
                [0.0, 0.0, 1.0],
            ]
        )
        self._mj_to_cv = np.diag([1.0, -1.0, -1.0])

        os.makedirs(out_dir, exist_ok=True)
        for i in range(num_drones):
            d_dir = os.path.join(out_dir, f"drone_{i}")
            os.makedirs(os.path.join(d_dir, "rgb"), exist_ok=True)
            os.makedirs(os.path.join(d_dir, "depth"), exist_ok=True)
            with open(
                os.path.join(d_dir, "intrinsics.json"), "w", encoding="utf-8"
            ) as f:
                json.dump(
                    {
                        "width": cam_spec.width,
                        "height": cam_spec.height,
                        "K": self.K.tolist(),
                    },
                    f,
                    indent=2,
                )

    def capture_step(
        self, data: mujoco.MjData, step_idx: int, drones: List[DroneState]
    ) -> None:
        ts = step_idx * self.model.opt.timestep
        for d in drones:
            if not d.active:
                continue
            cam_name = (
                f"cam_mono_{d.id}"
                if self.cam_spec.sensor_type == "mono"
                else f"cam_left_{d.id}"
            )
            cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)

            self.rgb_renderer.update_scene(data, camera=cam_name)
            rgb = self.rgb_renderer.render()
            self.depth_renderer.update_scene(data, camera=cam_name)
            depth_m = self.depth_renderer.render()

            T_wc = np.eye(4)
            T_wc[:3, :3] = data.cam_xmat[cam_id].reshape(3, 3) @ self._mj_to_cv
            T_wc[:3, 3] = data.cam_xpos[cam_id]

            d_dir = os.path.join(self.out_dir, f"drone_{d.id}")
            tag = f"{step_idx:06d}"
            iio.imwrite(os.path.join(d_dir, "rgb", f"{tag}.png"), rgb)
            iio.imwrite(
                os.path.join(d_dir, "depth", f"{tag}.png"),
                np.clip(depth_m * 1000.0, 0, 65535).astype(np.uint16),
            )
            with open(os.path.join(d_dir, "poses_se3.txt"), "a", encoding="utf-8") as f:
                f.write(
                    f"{ts:.4f} {tag} "
                    + " ".join(f"{v:.6f}" for v in T_wc.ravel())
                    + "\n"
                )

    def close(self) -> None:
        self.rgb_renderer.close()
        self.depth_renderer.close()


def build_swarm_xml(
    num_drones: int,
    cam_spec: CameraSpec,
    scenario: str = "urban",
    custom_mesh_path: Optional[str] = None,
    arena_size: float = 16.0,
) -> Tuple[str, np.ndarray]:
    """
    Generates the MuJoCo MJCF XML model and returns `(xml_str, charging_stations_xy)`.
    In the `charging` scenario, two glowing ground charging pads are placed at
    `(-4.8, 0.0)` and `(4.8, 0.0)` alongside two obstacles.
    """
    half = arena_size * 0.5
    assets, obstacles = "", ""
    stations = np.empty((0, 2), dtype=np.float64)

    if scenario == "urban":
        obstacles = """
        <geom name="bldg_1" type="box" pos="-3.5 3.0 1.4" size="1.4 1.4 1.4" rgba="0.35 0.38 0.45 1"/>
        <geom name="bldg_2" type="box" pos="3.5 -3.0 1.6" size="1.5 1.2 1.6" rgba="0.35 0.38 0.45 1"/>
        <geom name="bldg_3" type="box" pos="3.2 3.5 1.2" size="1.2 1.2 1.2" rgba="0.4 0.35 0.35 1"/>
        <geom name="bldg_4" type="box" pos="-3.8 -3.5 1.5" size="1.1 1.6 1.5" rgba="0.35 0.42 0.4 1"/>
        <geom name="central_monument" type="cylinder" pos="0 0 1.1" size="0.9 1.1" rgba="0.45 0.45 0.5 1"/>
        """
    elif scenario == "ruins":
        obstacles = """
        <geom name="wall_1" type="box" pos="-2.5 0.0 0.9" size="0.3 3.2 0.9" rgba="0.5 0.42 0.35 1"/>
        <geom name="wall_2" type="box" pos="2.5 1.5 0.8" size="2.2 0.3 0.8" rgba="0.5 0.42 0.35 1"/>
        <geom name="rubble_1" type="box" pos="1.0 -3.0 0.6" size="1.5 1.0 0.6" euler="15 20 35" rgba="0.45 0.4 0.35 1"/>
        <geom name="pillar_1" type="cylinder" pos="-4.5 4.0 1.5" size="0.6 1.5" rgba="0.55 0.5 0.45 1"/>
        <geom name="pillar_2" type="cylinder" pos="4.5 -4.0 1.3" size="0.7 1.3" rgba="0.55 0.5 0.45 1"/>
        """
    elif scenario == "charging":
        stations = np.array([[-4.8, 0.0], [4.8, 0.0]], dtype=np.float64)
        obstacles = """
        <!-- Two obstacles separating the north and south sectors -->
        <geom name="obs_north" type="box" pos="0.0 3.6 1.2" size="1.3 1.0 1.2" rgba="0.35 0.38 0.45 1"/>
        <geom name="obs_south" type="box" pos="0.0 -3.6 1.2" size="1.3 1.0 1.2" rgba="0.35 0.38 0.45 1"/>
        <!-- Charging Station 0 (West Pad) -->
        <geom name="pad_0_outer" type="cylinder" pos="-4.8 0.0 0.01" size="0.90 0.01" rgba="0.1 0.95 0.45 0.85" contype="0" conaffinity="0"/>
        <geom name="pad_0_inner" type="cylinder" pos="-4.8 0.0 0.015" size="0.45 0.01" rgba="0.9 1.0 0.95 0.95" contype="0" conaffinity="0"/>
        <!-- Charging Station 1 (East Pad) -->
        <geom name="pad_1_outer" type="cylinder" pos="4.8 0.0 0.01" size="0.90 0.01" rgba="0.1 0.95 0.45 0.85" contype="0" conaffinity="0"/>
        <geom name="pad_1_inner" type="cylinder" pos="4.8 0.0 0.015" size="0.45 0.01" rgba="0.9 1.0 0.95 0.95" contype="0" conaffinity="0"/>
        """
    elif (
        scenario == "custom_mesh"
        and custom_mesh_path
        and os.path.exists(custom_mesh_path)
    ):
        assets = (
            f'<mesh name="custom_env_mesh" file="{custom_mesh_path}" scale="1 1 1"/>'
        )
        obstacles = '<geom name="custom_3d_map" type="mesh" mesh="custom_env_mesh" pos="0 0 0" rgba="0.5 0.5 0.55 1"/>'

    bodies, actuators = [], []
    palette = [
        "0.2 0.85 0.35 1",
        "0.2 0.65 1.0 1",
        "1.0 0.6 0.15 1",
        "0.95 0.25 0.8 1",
        "0.15 0.9 0.9 1",
        "0.95 0.9 0.2 1",
    ]
    for i in range(num_drones):
        ang = 2.0 * np.pi * i / num_drones
        sx, sy = 6.2 * np.cos(ang), 6.2 * np.sin(ang)
        col = palette[i % len(palette)]
        if cam_spec.sensor_type == "mono":
            cams = (
                f'<camera name="cam_mono_{i}" pos="0 0 -0.06" fovy="{cam_spec.fovy}"/>'
            )
        else:
            hb = cam_spec.baseline * 0.5
            cams = f'<camera name="cam_left_{i}" pos="{-hb:.3f} 0 -0.06" fovy="{cam_spec.fovy}"/><camera name="cam_right_{i}" pos="{hb:.3f} 0 -0.06" fovy="{cam_spec.fovy}"/>'
        bodies.append(f"""
        <body name="drone_{i}" pos="{sx:.2f} {sy:.2f} 2.4">
            <freejoint name="joint_{i}"/>
            <geom name="core_{i}" type="box" size="0.22 0.22 0.06" rgba="{col}" mass="1.0"/>
            {cams}
        </body>
        <body name="target_vis_{i}" mocap="true" pos="{sx:.2f} {sy:.2f} 0.05">
            <geom type="sphere" size="0.16" rgba="{col}" contype="0" conaffinity="0"/>
        </body>""")
        actuators.append(
            f"""
        <general name="fx_{i}" joint="joint_{i}" gear="1 0 0 0 0 0" ctrlrange="-25 25"/>
        <general name="fy_{i}" joint="joint_{i}" gear="0 1 0 0 0 0" ctrlrange="-25 25"/>
        <general name="fz_{i}" joint="joint_{i}" gear="0 0 1 0 0 0" ctrlrange="-25 35"/>"""
        )

    xml = f"""
    <mujoco model="drone_swarm">
        <option timestep="0.01" gravity="0 0 -9.81" integrator="RK4"/>
        <asset>
            <texture name="grid" type="2d" builtin="checker" width="512" height="512" rgb1="0.16 0.18 0.21" rgb2="0.20 0.22 0.26"/>
            <material name="floor_mat" texture="grid" texrepeat="8 8" reflectance="0.05"/>
            {assets}
        </asset>
        <worldbody>
            <light pos="0 0 18" dir="0 0 -1" directional="true"/>
            <geom name="floor" type="plane" size="{half} {half} 0.1" material="floor_mat"/>
            <geom type="box" pos="0 {half} 0.6" size="{half} 0.1 0.6" rgba="0.4 0.4 0.4 0.4"/>
            <geom type="box" pos="0 {-half} 0.6" size="{half} 0.1 0.6" rgba="0.4 0.4 0.4 0.4"/>
            <geom type="box" pos="{half} 0 0.6" size="0.1 {half} 0.6" rgba="0.4 0.4 0.4 0.4"/>
            <geom type="box" pos="{-half} 0 0.6" size="0.1 {half} 0.6" rgba="0.4 0.4 0.4 0.4"/>
            {obstacles}{"".join(bodies)}
        </worldbody>
        <actuator>{"".join(actuators)}</actuator>
    </mujoco>"""
    return xml, stations


def save_coverage_report(grid: CoverageGrid, drones: List[DroneState]) -> None:
    """Saves a PNG showing individual and combined coverage maps + charging stations."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        res, half = grid.resolution, grid.size * 0.5
        obs_2d = (~grid.free_cells).reshape(res, res)
        fig, axes = plt.subplots(
            1, len(drones) + 1, figsize=(3.6 * (len(drones) + 1), 3.8), dpi=140
        )
        for i, d in enumerate(drones):
            cov = grid.drone_sole_visited[i].reshape(res, res).astype(float)
            cov[obs_2d] = -0.5
            axes[i].imshow(
                cov,
                cmap="viridis",
                vmin=-0.5,
                vmax=1.0,
                origin="lower",
                extent=[-half, half, -half, half],
            )
            if len(grid.charging_stations) > 0:
                axes[i].plot(
                    grid.charging_stations[:, 0],
                    grid.charging_stations[:, 1],
                    "gs",
                    markersize=8,
                )
            axes[i].plot(d.pos[0], d.pos[1], "ro", markersize=7)
            axes[i].set_title(
                f"Drone {i} (Bat: {d.battery:.0f}%)\n({grid.get_drone_coverage_pct(i):.1f}%)"
            )

        comb = grid.combined_visited_map.reshape(res, res).astype(float)
        comb[obs_2d] = -0.5
        axes[-1].imshow(
            comb,
            cmap="plasma",
            vmin=-0.5,
            vmax=1.0,
            origin="lower",
            extent=[-half, half, -half, half],
        )
        if len(grid.charging_stations) > 0:
            axes[-1].plot(
                grid.charging_stations[:, 0],
                grid.charging_stations[:, 1],
                "gs",
                markersize=9,
                label="Charger",
            )
        for d in drones:
            axes[-1].plot(d.pos[0], d.pos[1], "wo", markersize=6)
        axes[-1].set_title(f"COMBINED\n({grid.combined_coverage_pct:.1f}%)")
        plt.tight_layout()
        plt.savefig("coverage_maps_comparison.png")
        plt.close(fig)
        print("-> Saved 'coverage_maps_comparison.png'")
    except ImportError:
        pass


def run_simulation(
    algo_fn: Callable[[DroneState, List[DroneState], CoverageGrid], np.ndarray],
    algo_name: str,
    mode: str = "headless",
    sensor_type: str = "stereo",
    scenario: str = "flat",
    custom_mesh: Optional[str] = None,
    num_drones: int = 4,
    max_steps: int = 2500,
    show_assignment: bool = True,
    save_slam_data: bool = False,
    slam_interval: int = 25,
    enable_battery: bool = True,
) -> dict:
    """Runs the MuJoCo physics loop with battery drain, charging stations, and KPI logging."""
    cam_spec = CameraSpec(sensor_type=sensor_type)
    xml_str, stations = build_swarm_xml(num_drones, cam_spec, scenario, custom_mesh)
    model = mujoco.MjModel.from_xml_string(xml_str)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    body_ids = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"drone_{i}")
        for i in range(num_drones)
    ]
    geom_ids = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"core_{i}")
        for i in range(num_drones)
    ]

    grid = CoverageGrid(num_drones=num_drones)
    grid.charging_stations = stations
    grid.mark_obstacles(model, data)

    drones = [
        DroneState(
            i,
            data.qpos[i * 7 : i * 7 + 3].copy(),
            np.zeros(3),
            target=data.qpos[i * 7 : i * 7 + 3].copy(),
        )
        for i in range(num_drones)
    ]
    slam = (
        MultiDroneSlamExtractor(model, cam_spec, num_drones) if save_slam_data else None
    )
    metrics = SwarmMetricsTracker(dt=model.opt.timestep)
    kp, kd, mass = 6.0, 4.5, 1.2

    def update_battery_and_charging() -> None:
        """Simulates thrust-based battery drain in flight and fast recharging on pads."""
        dt = model.opt.timestep
        has_pads = len(grid.charging_stations) > 0

        for d in drones:
            if not d.active:
                d.is_charging = False
                continue

            # Check if drone has touched down on any charging station pad
            on_pad = False
            if has_pads and d.pos[2] <= 0.45:
                pad_dists = np.linalg.norm(grid.charging_stations - d.pos[:2], axis=1)
                on_pad = bool(np.min(pad_dists) <= grid.pad_radius)

            if on_pad and d.battery < 100.0:
                if not d.is_charging:
                    d.charge_visits += 1
                d.is_charging = True
                # Recharge at 35% per second (0.35% per 0.01s step)
                d.battery = min(100.0, d.battery + 0.35)
                d.charge_time += dt
            else:
                d.is_charging = False
                # Base avionics/hover drain (4.5%/s) + velocity-dependent propulsion drain (up to 2.5%/s)
                speed_xy = float(np.linalg.norm(d.vel[:2]))
                drain = 0.045 + 0.025 * min(1.0, speed_xy / 7.0)
                d.battery = max(0.0, d.battery - drain)
                d.energy_used += drain

                if d.battery <= 0.0:
                    d.active = False
                    d.depleted = True
                    model.geom_rgba[geom_ids[d.id]] = [0.8, 0.1, 0.1, 0.5]
                    print(
                        f"\n[BATTERY EMPTY] Drone {d.id} ran out of battery and grounded!"
                    )

    def step_fn(step: int) -> bool:
        for i, d in enumerate(drones):
            d.pos[:] = data.qpos[i * 7 : i * 7 + 3]
            d.vel[:] = data.qvel[i * 6 : i * 6 + 3]
            data.qpos[i * 7 + 3 : i * 7 + 7] = (1.0, 0.0, 0.0, 0.0)
            data.qvel[i * 6 + 3 : i * 6 + 6] = (0.0, 0.0, 0.0)

        # Inject random hardware failure at step 500 ONLY outside the `charging` scenario
        if (
            scenario != "charging"
            and step == 500
            and num_drones > 1
            and drones[0].active
        ):
            drones[0].active = False
            model.geom_rgba[geom_ids[0]] = [0.8, 0.1, 0.1, 0.4]
            print(f"\n[EVENT @ Step {step}] Drone 0 FAILED! Redistributing workload...")

        if enable_battery:
            update_battery_and_charging()

        if step % 4 == 0:
            grid.update_from_cameras(drones, body_ids, cam_spec, model, data)

        if step % 10 == 0:
            all_pos = np.array([d.pos for d in drones])
            for i, d in enumerate(drones):
                if not d.active:
                    d.target[:] = (d.pos[0], d.pos[1], 0.15)
                    continue
                dists = np.linalg.norm(all_pos - d.pos, axis=1)
                peers = [
                    drones[j]
                    for j in range(num_drones)
                    if j != i and dists[j] <= d.comm_radius
                ]
                tgt = algo_fn(d, peers, grid)
                d.target[:] = tgt if tgt is not None else (d.pos[0], d.pos[1], 2.4)
                data.mocap_pos[i] = (d.target[0], d.target[1], 0.05)
            grid.update_area_assignments(drones, algo_name)

        if slam and step % slam_interval == 0:
            slam.capture_step(data, step, drones)

        for i, d in enumerate(drones):
            acc = np.clip((d.target - d.pos) * kp, -7.0, 7.0) - kd * d.vel
            data.ctrl[i * 3 : i * 3 + 3] = [
                mass * acc[0],
                mass * acc[1],
                mass * (acc[2] + 9.81),
            ]

        mujoco.mj_step(model, data)

        cov_pct = grid.combined_coverage_pct
        metrics.update(step, cov_pct)

        if step % 250 == 0:
            status = " | ".join(
                f"D{i}:{grid.get_drone_coverage_pct(i):.0f}%(B:{drones[i].battery:.0f}%)"
                for i in range(num_drones)
            )
            print(f"Step {step:04d} | Combined: {cov_pct:5.1f}% | [{status}]")

        # Stop early if 99.8% covered or if all drones are inactive
        any_active = any(d.active for d in drones)
        return (cov_pct >= 99.8) or (not any_active)

    steps_executed = 0
    if mode == "visual":
        from mujoco import viewer

        with viewer.launch_passive(model, data) as v:
            while v.is_running() and steps_executed < max_steps:
                t0 = time.perf_counter()
                done = step_fn(steps_executed)
                if show_assignment:
                    render_live_assignment_overlay(v, grid)
                v.sync()
                steps_executed += 1
                if done:
                    break
                dt = time.perf_counter() - t0
                if dt < model.opt.timestep:
                    time.sleep(model.opt.timestep - dt)
    else:
        for step in range(max_steps):
            steps_executed = step + 1
            if step_fn(step):
                break

    if slam:
        slam.close()
    save_coverage_report(grid, drones)
    return metrics.summarize_and_save(steps_executed, grid, drones, algo_name, scenario)

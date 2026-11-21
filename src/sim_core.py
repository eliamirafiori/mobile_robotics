"""
sim_core.py — Standalone MuJoCo 3D Drone Swarm Simulation Core
==============================================================
Ready-to-use physics, sensor, battery, map/dataset, and benchmarking engine.
Keep your research focused on high-level planning in `algorithms.py` while
treating this file as a stable, decoupled testbed.

Key abstractions:
  - `ObstacleSpec` / `MapSpec` : Declarative map & scenario model with JSON
    serialization, presets (`flat`, `urban`, `ruins`), random map generation,
    and dataset save/load helpers.
  - `SimConfig`                : Decoupled runtime configuration (number of
    drones, battery management toggle, failure injection, visualization, I/O).
  - `SwarmSimulator`           : Self-contained MuJoCo execution engine.
  - `run_benchmark_suite`      : Runs multiple algorithms across N maps in both
    `no_battery` and `with_battery` scenarios and exports CSV/JSON reports.
"""

from __future__ import annotations

import csv
import json
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import imageio.v3 as iio
import mujoco
import numpy as np

# =====================================================================
# 1. MAP & SCENARIO SPECIFICATION (PRESETS, RANDOM MAPS, DATASETS)
# =====================================================================


@dataclass
class ObstacleSpec:
    """Declarative 3D static obstacle placed in the MuJoCo worldbody."""

    shape: str  # "box" or "cylinder"
    pos: List[float]  # [x, y, z] center in meters
    size: List[float]  # box: [hx, hy, hz]; cylinder: [radius, hz]
    euler: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    rgba: List[float] = field(default_factory=lambda: [0.38, 0.40, 0.46, 1.0])
    name: str = ""

    def footprint_radius(self) -> float:
        """Conservative 2D horizontal bounding radius for placement checks."""
        if self.shape == "cylinder":
            return float(self.size[0])
        return float(np.hypot(self.size[0], self.size[1]))

    def to_xml(self, fallback_name: str) -> str:
        geom_name = self.name or fallback_name
        pos_str = " ".join(f"{v:.3f}" for v in self.pos)
        size_str = " ".join(f"{v:.3f}" for v in self.size)
        rgba_str = " ".join(f"{v:.3f}" for v in self.rgba)
        euler_attr = (
            f' euler="{" ".join(f"{v:.2f}" for v in self.euler)}"'
            if any(abs(v) > 1e-6 for v in self.euler)
            else ""
        )
        return (
            f'<geom name="{geom_name}" type="{self.shape}" '
            f'pos="{pos_str}" size="{size_str}"{euler_attr} rgba="{rgba_str}"/>'
        )


@dataclass
class MapSpec:
    """
    Serializable map definition decoupled from battery/algorithm settings.
    Pass a `MapSpec` directly into `SwarmSimulator` or `run_benchmark_suite`.
    """

    name: str = "flat"
    size: float = 16.0  # Square arena side length in meters
    resolution: int = 40  # Discretized grid cells per axis
    obstacles: List[ObstacleSpec] = field(default_factory=list)
    charging_stations: List[List[float]] = field(default_factory=list)
    mesh_file: Optional[str] = None
    spawn_points: Optional[List[List[float]]] = None
    pad_radius: float = 0.85

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "size": float(self.size),
            "resolution": int(self.resolution),
            "obstacles": [asdict(o) for o in self.obstacles],
            "charging_stations": [
                [float(p[0]), float(p[1])] for p in self.charging_stations
            ],
            "mesh_file": self.mesh_file,
            "spawn_points": (
                [[float(p[0]), float(p[1])] for p in self.spawn_points]
                if self.spawn_points is not None
                else None
            ),
            "pad_radius": float(self.pad_radius),
        }

    @classmethod
    def from_dict(cls, data: dict) -> MapSpec:
        obstacles = [ObstacleSpec(**o) for o in data.get("obstacles", [])]
        return cls(
            name=str(data.get("name", "unnamed_map")),
            size=float(data.get("size", 16.0)),
            resolution=int(data.get("resolution", 40)),
            obstacles=obstacles,
            charging_stations=[
                [float(p[0]), float(p[1])] for p in data.get("charging_stations", [])
            ],
            mesh_file=data.get("mesh_file"),
            spawn_points=data.get("spawn_points"),
            pad_radius=float(data.get("pad_radius", 0.85)),
        )

    def save_json(self, path: str) -> str:
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        return path

    @classmethod
    def load_json(cls, path: str) -> MapSpec:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError(f"Expected a map JSON object in '{path}'.")
        return cls.from_dict(data)

    @classmethod
    def from_preset(
        cls,
        preset: str,
        size: float = 16.0,
        resolution: int = 40,
        num_stations: int = 2,
        mesh_file: Optional[str] = None,
    ) -> MapSpec:
        """Creates a built-in map scaled cleanly to `size`."""
        scale = size / 16.0
        obstacles: List[ObstacleSpec] = []

        if preset == "urban":
            obstacles = [
                ObstacleSpec(
                    "box",
                    [-3.5 * scale, 3.0 * scale, 1.4],
                    [1.4 * scale, 1.4 * scale, 1.4],
                    name="bldg_1",
                ),
                ObstacleSpec(
                    "box",
                    [3.5 * scale, -3.0 * scale, 1.6],
                    [1.5 * scale, 1.2 * scale, 1.6],
                    name="bldg_2",
                ),
                ObstacleSpec(
                    "box",
                    [3.2 * scale, 3.5 * scale, 1.2],
                    [1.2 * scale, 1.2 * scale, 1.2],
                    rgba=[0.42, 0.35, 0.35, 1.0],
                    name="bldg_3",
                ),
                ObstacleSpec(
                    "box",
                    [-3.8 * scale, -3.5 * scale, 1.5],
                    [1.1 * scale, 1.6 * scale, 1.5],
                    rgba=[0.35, 0.42, 0.40, 1.0],
                    name="bldg_4",
                ),
                ObstacleSpec(
                    "cylinder",
                    [0.0, 0.0, 1.1],
                    [0.9 * scale, 1.1],
                    rgba=[0.45, 0.45, 0.50, 1.0],
                    name="monument",
                ),
            ]
        elif preset == "ruins":
            obstacles = [
                ObstacleSpec(
                    "box",
                    [-2.5 * scale, 0.0, 0.9],
                    [0.3 * scale, 3.2 * scale, 0.9],
                    rgba=[0.50, 0.42, 0.35, 1.0],
                    name="wall_1",
                ),
                ObstacleSpec(
                    "box",
                    [2.5 * scale, 1.5 * scale, 0.8],
                    [2.2 * scale, 0.3 * scale, 0.8],
                    rgba=[0.50, 0.42, 0.35, 1.0],
                    name="wall_2",
                ),
                ObstacleSpec(
                    "box",
                    [1.0 * scale, -3.0 * scale, 0.6],
                    [1.5 * scale, 1.0 * scale, 0.6],
                    euler=[15.0, 20.0, 35.0],
                    rgba=[0.45, 0.40, 0.35, 1.0],
                    name="rubble_1",
                ),
                ObstacleSpec(
                    "cylinder",
                    [-4.5 * scale, 4.0 * scale, 1.5],
                    [0.6 * scale, 1.5],
                    rgba=[0.55, 0.50, 0.45, 1.0],
                    name="pillar_1",
                ),
                ObstacleSpec(
                    "cylinder",
                    [4.5 * scale, -4.0 * scale, 1.3],
                    [0.7 * scale, 1.3],
                    rgba=[0.55, 0.50, 0.45, 1.0],
                    name="pillar_2",
                ),
            ]
        elif preset == "corridors":
            obstacles = [
                ObstacleSpec(
                    "box",
                    [0.0, 3.6 * scale, 1.2],
                    [1.3 * scale, 1.0 * scale, 1.2],
                    name="obs_north",
                ),
                ObstacleSpec(
                    "box",
                    [0.0, -3.6 * scale, 1.2],
                    [1.3 * scale, 1.0 * scale, 1.2],
                    name="obs_south",
                ),
            ]
        elif preset == "custom_mesh":
            if not mesh_file or not os.path.exists(mesh_file):
                raise FileNotFoundError(f"Custom mesh file not found: '{mesh_file}'")
        elif preset != "flat":
            raise ValueError(
                f"Unknown preset '{preset}'. Choose from: flat, urban, ruins, corridors, custom_mesh."
            )

        stations = place_charging_stations(
            num_stations=num_stations,
            arena_size=size,
            obstacles=obstacles,
        )
        return cls(
            name=preset,
            size=size,
            resolution=resolution,
            obstacles=obstacles,
            charging_stations=stations.tolist(),
            mesh_file=mesh_file if preset == "custom_mesh" else None,
        )

    @classmethod
    def generate_random(
        cls,
        size: float = 16.0,
        resolution: int = 40,
        num_obstacles: int = 5,
        num_stations: int = 2,
        seed: Optional[int] = None,
        name: Optional[str] = None,
    ) -> MapSpec:
        """
        Procedurally generates a valid random map with non-overlapping boxes and
        cylinders while preserving clear landing zones around charging stations.
        """
        rng = np.random.default_rng(seed)
        half = size * 0.5
        pad_radius = 0.85

        # 1. Place charging stations first so obstacles never overlap them
        stations = place_charging_stations(
            num_stations=num_stations,
            arena_size=size,
            obstacles=[],
            pad_radius=pad_radius,
        )

        # 2. Sample non-overlapping 3D obstacles inside the arena interior
        obstacles: List[ObstacleSpec] = []
        scale = size / 16.0
        min_wall_margin = max(1.6, 1.8 * scale)

        for obs_idx in range(num_obstacles):
            for _attempt in range(80):
                shape = "box" if rng.random() < 0.7 else "cylinder"
                height = float(rng.uniform(0.9, 1.8))
                if shape == "box":
                    hx = float(rng.uniform(0.6, 1.6) * scale)
                    hy = float(rng.uniform(0.6, 1.6) * scale)
                    obs_size = [hx, hy, height]
                    bound_r = float(np.hypot(hx, hy))
                else:
                    rad = float(rng.uniform(0.5, 1.2) * scale)
                    obs_size = [rad, height]
                    bound_r = rad

                limit = half - min_wall_margin - bound_r
                if limit <= 0.5:
                    continue

                cx = float(rng.uniform(-limit, limit))
                cy = float(rng.uniform(-limit, limit))

                # Keep clear of charging pads
                if len(stations) > 0:
                    pad_dists = np.linalg.norm(stations - np.array([cx, cy]), axis=1)
                    if np.min(pad_dists) < (bound_r + pad_radius + 1.1):
                        continue

                # Keep clear of previously placed obstacles
                overlaps = False
                for existing in obstacles:
                    ex_dist = float(
                        np.hypot(cx - existing.pos[0], cy - existing.pos[1])
                    )
                    if ex_dist < (bound_r + existing.footprint_radius() + 1.0):
                        overlaps = True
                        break
                if overlaps:
                    continue

                tone = float(rng.uniform(0.32, 0.52))
                obstacles.append(
                    ObstacleSpec(
                        shape=shape,
                        pos=[cx, cy, height],
                        size=obs_size,
                        rgba=[tone, tone + 0.03, tone + 0.07, 1.0],
                        name=f"rand_obs_{obs_idx}",
                    )
                )
                break

        map_name = name or f"random_s{seed if seed is not None else 'none'}"
        return cls(
            name=map_name,
            size=size,
            resolution=resolution,
            obstacles=obstacles,
            charging_stations=stations.tolist(),
            pad_radius=pad_radius,
        )

    @classmethod
    def generate_dataset(
        cls,
        out_dir: str,
        num_maps: int = 5,
        size: float = 16.0,
        resolution: int = 40,
        num_obstacles_range: Tuple[int, int] = (3, 7),
        num_stations: int = 2,
        base_seed: int = 0,
    ) -> List[MapSpec]:
        """Generates `num_maps` random maps, saves them as JSON in `out_dir`, and returns them."""
        os.makedirs(out_dir, exist_ok=True)
        rng = np.random.default_rng(base_seed)
        maps: List[MapSpec] = []

        low_obs, high_obs = num_obstacles_range
        for idx in range(num_maps):
            seed = base_seed + idx
            n_obs = int(rng.integers(low_obs, high_obs + 1))
            m = cls.generate_random(
                size=size,
                resolution=resolution,
                num_obstacles=n_obs,
                num_stations=num_stations,
                seed=seed,
                name=f"map_{idx:03d}",
            )
            m.save_json(os.path.join(out_dir, f"map_{idx:03d}.json"))
            maps.append(m)

        return maps

    @classmethod
    def load_dataset(cls, path: str) -> List[MapSpec]:
        """
        Loads a dataset of maps from either:
          - A directory containing `.json` map files (sorted by filename), OR
          - A single `.json` file containing a single map object, a list of maps,
            or `{"maps": [...]}`.
        """
        if os.path.isdir(path):
            files = sorted(
                os.path.join(path, fn)
                for fn in os.listdir(path)
                if fn.lower().endswith(".json")
            )
            if not files:
                raise FileNotFoundError(
                    f"No .json map files found in directory '{path}'."
                )
            return [cls.load_json(fp) for fp in files]

        if os.path.isfile(path):
            with open(path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            if isinstance(raw, list):
                return [cls.from_dict(item) for item in raw]
            if isinstance(raw, dict) and "maps" in raw:
                return [cls.from_dict(item) for item in raw["maps"]]
            if isinstance(raw, dict):
                return [cls.from_dict(raw)]

        raise FileNotFoundError(f"Map dataset path not found: '{path}'")


def place_charging_stations(
    num_stations: int,
    arena_size: float,
    obstacles: Sequence[ObstacleSpec] = (),
    pad_radius: float = 0.85,
) -> np.ndarray:
    """
    Places charging pads evenly across the arena while avoiding static obstacles.
    Returns an `(N, 2)` float64 array of `[x, y]` pad centers.
    """
    if num_stations <= 0:
        return np.empty((0, 2), dtype=np.float64)

    half = arena_size * 0.5
    margin = pad_radius + 0.5
    max_reach = half - margin
    if max_reach <= 0.0:
        raise ValueError(f"Arena size {arena_size}m is too small for charging pads.")

    if num_stations == 1:
        candidates = [np.array([0.0, 0.0])]
    else:
        min_spacing = 2.0 * pad_radius + 0.4
        reach = min(
            max(arena_size * 0.30, min_spacing * (num_stations - 1) * 0.5), max_reach
        )
        xs = np.linspace(-reach, reach, num_stations)
        candidates = [np.array([float(x), 0.0]) for x in xs]

    # Nudge any pad that intersects an obstacle along Y until clear
    placed: List[np.ndarray] = []
    for pt in candidates:
        chosen = pt.copy()
        for shift_y in (0.0, 1.5, -1.5, 3.0, -3.0, 4.5, -4.5):
            test = np.array([pt[0], np.clip(pt[1] + shift_y, -max_reach, max_reach)])
            blocked = any(
                np.hypot(test[0] - o.pos[0], test[1] - o.pos[1])
                < (o.footprint_radius() + pad_radius + 0.4)
                for o in obstacles
            )
            if not blocked:
                chosen = test
                break
        placed.append(chosen)

    return np.asarray(placed, dtype=np.float64)


# =====================================================================
# 2. SIMULATION CONFIGURATION & RUNTIME STATE STRUCTURES
# =====================================================================


@dataclass
class SimConfig:
    """Decoupled runtime settings controlling any simulation execution."""

    num_drones: int = 4
    max_steps: int = 2500
    battery_enabled: bool = False  # Toggle battery drain & charging scenario
    failure_step: Optional[int] = None  # Optional forced failure step (e.g. 500)
    sensor_type: str = "stereo"  # "mono" or "stereo"
    comm_radius: Optional[float] = (
        None  # None -> auto-scales to max(9.0, 0.6 * map.size)
    )
    cruise_altitude: float = 2.4
    battery_drain_base: float = 0.045  # % per physics step (hover/avionics)
    battery_drain_vel: float = 0.025  # extra % per physics step at max speed
    battery_charge_rate: float = 0.35  # % per physics step on pad (35%/sec)
    scale_battery_with_size: bool = True  # Scales endurance for large arenas (>16m)
    mode: str = "headless"  # "headless" or "visual"
    show_assignment: bool = True
    save_slam_data: bool = False
    slam_interval: int = 25
    save_plots: bool = True
    save_metrics: bool = True
    output_dir: str = "results"
    verbose: bool = True


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
    active: bool = True  # False on failure or empty battery
    comm_radius: float = 9.0  # P2P radio range (m)
    target: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))

    # --- Battery & Charging Telemetry ---
    battery: float = 100.0  # State of Charge: 0.0% to 100.0%
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
    battery_enabled: bool = False
    pad_radius: float = 0.85

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
        # Local CRDT belief map per drone: 1.0 = unknown, 0.05 = well-scanned
        self.drone_belief_uncertainty = np.ones(
            (self.num_drones, self.num_cells), dtype=np.float32
        )
        # Live territory owner ID (-1 = obstacle/unassigned)
        self.cell_owner = np.full(self.num_cells, -1, dtype=np.int32)
        # Traversable ground mask (False where static 3D obstacles sit)
        self.free_cells = np.ones(self.num_cells, dtype=bool)
        self.obstacle_centers = np.empty((0, 2), dtype=np.float64)
        self.total_free_cells = self.num_cells

        # Active charging stations [N, 2] (empty when battery_enabled is False)
        self.charging_stations = np.empty((0, 2), dtype=np.float64)

    def mark_obstacles(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        """Detects static worldbody obstacles via downward rays from z = 10m."""
        origin = np.array([0.0, 0.0, 10.0], dtype=np.float64)
        direction = np.array([0.0, 0.0, -1.0], dtype=np.float64)
        geom_out = np.zeros(1, dtype=np.int32)

        for idx, (cx, cy) in enumerate(self.cell_centers):
            origin[0], origin[1] = cx, cy
            hit = mujoco.mj_ray(model, data, origin, direction, None, 1, -1, geom_out)
            geom_id = int(geom_out[0])
            hit_body = int(model.geom_bodyid[geom_id]) if geom_id >= 0 else -1
            # Treat only solid static worldbody geoms (body 0, height > 0.25m) as obstacles
            if 0.0 < hit < 9.75 and hit_body == 0:
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
        """Projects camera frustums, checks 3D line-of-sight occlusion, and merges P2P maps."""
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
# 3. PERFORMANCE METRICS TRACKER (COVERAGE TIME & BATTERY MANAGEMENT)
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

    def build_report(
        self,
        total_steps: int,
        grid: CoverageGrid,
        drones: List[DroneState],
        algo_name: str,
        map_spec: MapSpec,
        battery_enabled: bool,
    ) -> dict:
        total_energy_used = sum(d.energy_used for d in drones)
        total_charge_time = sum(d.charge_time for d in drones)
        total_charge_visits = sum(d.charge_visits for d in drones)
        depleted_count = sum(1 for d in drones if d.depleted)
        active_count = sum(1 for d in drones if d.active)
        avg_final_battery = float(np.mean([d.battery for d in drones]))
        min_final_battery = float(np.min([d.battery for d in drones]))

        return {
            "algorithm": algo_name,
            "map_name": map_spec.name,
            "scenario_mode": "with_battery" if battery_enabled else "no_battery",
            "battery_enabled": battery_enabled,
            "num_drones": len(drones),
            "grid_size_m": map_spec.size,
            "grid_resolution": map_spec.resolution,
            "num_obstacles": len(map_spec.obstacles),
            "num_charging_stations": int(len(grid.charging_stations)),
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

    @staticmethod
    def print_report(report: dict) -> None:
        cov = report["coverage_time_metrics"]
        bat = report["battery_management_metrics"]
        print("\n" + "=" * 68)
        print(
            f"METRICS | Algo: {report['algorithm'].upper()} | Map: {report['map_name']} "
            f"| Mode: {report['scenario_mode'].upper()}"
        )
        print(
            f"   Arena: {report['grid_size_m']:.1f}m ({report['grid_resolution']}x{report['grid_resolution']}) "
            f"| Obstacles: {report['num_obstacles']} | Charging pads: {report['num_charging_stations']}"
        )
        print("-" * 68)
        print("1. COVERAGE TIME:")
        for pct in (50, 80, 90, 95, 99):
            info = cov[f"time_to_{pct}_pct"]
            val = (
                f"{info['time_sec']}s (step {info['step']})" if info else "Not reached"
            )
            print(f"   - Time to {pct:3d}% coverage : {val}")
        print(
            f"   - Final coverage        : {cov['final_coverage_pct']:.1f}% "
            f"in {cov['total_sim_time_sec']:.2f}s ({cov['total_steps_run']} steps)"
        )
        print("2. BATTERY MANAGEMENT:")
        if report["battery_enabled"]:
            print(
                f"   - Total energy drained  : {bat['total_energy_drained_pct']:.1f}% (across swarm)"
            )
            print(
                f"   - Final battery (avg/min): {bat['avg_final_battery_pct']:.1f}% / {bat['min_final_battery_pct']:.1f}%"
            )
            print(
                f"   - Charging dockings     : {bat['total_charging_dockings']} visits ({bat['total_charging_time_sec']:.2f}s total)"
            )
            print(
                f"   - Empty-battery crashes : {bat['battery_depletion_failures']} | Surviving: {bat['surviving_drones']}"
            )
        else:
            print("   - Battery simulation disabled (no_battery scenario)")
        print("=" * 68)


# =====================================================================
# 4. XML SCENE COMPILER, VISUAL OVERLAY & SLAM EXTRACTOR
# =====================================================================

DRONE_PALETTE_RGBA = [
    [0.20, 0.85, 0.35, 0.30],
    [0.20, 0.65, 1.00, 0.30],
    [1.00, 0.60, 0.15, 0.30],
    [0.95, 0.25, 0.80, 0.30],
    [0.15, 0.90, 0.90, 0.30],
    [0.95, 0.90, 0.20, 0.30],
]


def compute_safe_spawns(
    num_drones: int, map_spec: MapSpec
) -> List[Tuple[float, float]]:
    """Returns `num_drones` valid 2D spawn coordinates guaranteed clear of walls & obstacles."""
    if map_spec.spawn_points and len(map_spec.spawn_points) >= num_drones:
        return [(float(p[0]), float(p[1])) for p in map_spec.spawn_points[:num_drones]]

    half = map_spec.size * 0.5
    base_r = half * 0.75
    limit = half - 1.0
    spawns: List[Tuple[float, float]] = []

    for i in range(num_drones):
        base_ang = 2.0 * np.pi * i / max(1, num_drones)
        chosen = (base_r * np.cos(base_ang), base_r * np.sin(base_ang))
        for r_scale in (1.0, 0.85, 0.65, 0.45, 0.25):
            r = min(limit, base_r * r_scale)
            found = False
            for ang_off in (0.0, 0.25, -0.25, 0.5, -0.5, 0.8, -0.8):
                sx = float(np.clip(r * np.cos(base_ang + ang_off), -limit, limit))
                sy = float(np.clip(r * np.sin(base_ang + ang_off), -limit, limit))
                blocked = any(
                    np.hypot(sx - o.pos[0], sy - o.pos[1])
                    < (o.footprint_radius() + 0.75)
                    for o in map_spec.obstacles
                )
                if not blocked:
                    chosen = (sx, sy)
                    found = True
                    break
            if found:
                break
        spawns.append(chosen)
    return spawns


def build_mujoco_xml(
    map_spec: MapSpec,
    num_drones: int,
    cam_spec: CameraSpec,
    active_stations: np.ndarray,
    cruise_altitude: float = 2.4,
) -> str:
    """Compiles a `MapSpec` + swarm layout into a valid MuJoCo MJCF XML string."""
    half = map_spec.size * 0.5
    assets_xml = ""
    obstacle_xml_parts = [
        obs.to_xml(f"obs_{i}") for i, obs in enumerate(map_spec.obstacles)
    ]

    if map_spec.mesh_file and os.path.exists(map_spec.mesh_file):
        assets_xml = (
            f'<mesh name="custom_env_mesh" file="{map_spec.mesh_file}" scale="1 1 1"/>'
        )
        obstacle_xml_parts.append(
            '<geom name="custom_3d_map" type="mesh" mesh="custom_env_mesh" pos="0 0 0" rgba="0.5 0.5 0.55 1"/>'
        )

    # Render visual-only glowing green pads whenever active charging stations exist
    for i, (px, py) in enumerate(active_stations):
        obstacle_xml_parts.append(
            f'<geom name="pad_{i}_outer" type="cylinder" pos="{px:.3f} {py:.3f} 0.01" '
            f'size="{map_spec.pad_radius:.2f} 0.01" rgba="0.10 0.95 0.45 0.85" contype="0" conaffinity="0"/>'
            f'<geom name="pad_{i}_inner" type="cylinder" pos="{px:.3f} {py:.3f} 0.015" '
            f'size="{map_spec.pad_radius * 0.5:.2f} 0.01" rgba="0.90 1.00 0.95 0.95" contype="0" conaffinity="0"/>'
        )

    spawns = compute_safe_spawns(num_drones, map_spec)
    bodies, actuators = [], []
    palette = [
        "0.20 0.85 0.35 1",
        "0.20 0.65 1.00 1",
        "1.00 0.60 0.15 1",
        "0.95 0.25 0.80 1",
        "0.15 0.90 0.90 1",
        "0.95 0.90 0.20 1",
    ]

    for i, (sx, sy) in enumerate(spawns):
        col = palette[i % len(palette)]
        if cam_spec.sensor_type == "mono":
            cams = (
                f'<camera name="cam_mono_{i}" pos="0 0 -0.06" fovy="{cam_spec.fovy}"/>'
            )
        else:
            hb = cam_spec.baseline * 0.5
            cams = (
                f'<camera name="cam_left_{i}" pos="{-hb:.3f} 0 -0.06" fovy="{cam_spec.fovy}"/>'
                f'<camera name="cam_right_{i}" pos="{hb:.3f} 0 -0.06" fovy="{cam_spec.fovy}"/>'
            )

        bodies.append(f"""
        <body name="drone_{i}" pos="{sx:.3f} {sy:.3f} {cruise_altitude:.2f}">
            <freejoint name="joint_{i}"/>
            <geom name="core_{i}" type="box" size="0.22 0.22 0.06" rgba="{col}" mass="1.0"/>
            {cams}
        </body>
        <body name="target_vis_{i}" mocap="true" pos="{sx:.3f} {sy:.3f} 0.05">
            <geom type="sphere" size="0.16" rgba="{col}" contype="0" conaffinity="0"/>
        </body>""")

        actuators.append(
            f"""
        <general name="fx_{i}" joint="joint_{i}" gear="1 0 0 0 0 0" ctrlrange="-25 25"/>
        <general name="fy_{i}" joint="joint_{i}" gear="0 1 0 0 0 0" ctrlrange="-25 25"/>
        <general name="fz_{i}" joint="joint_{i}" gear="0 0 1 0 0 0" ctrlrange="-25 35"/>"""
        )

    tex_rep = max(4, int(round(map_spec.size * 0.5)))
    return f"""
    <mujoco model="drone_swarm_{map_spec.name}">
        <option timestep="0.01" gravity="0 0 -9.81" integrator="RK4"/>
        <asset>
            <texture name="grid" type="2d" builtin="checker" width="512" height="512"
                     rgb1="0.16 0.18 0.21" rgb2="0.20 0.22 0.26"/>
            <material name="floor_mat" texture="grid" texrepeat="{tex_rep} {tex_rep}" reflectance="0.05"/>
            {assets_xml}
        </asset>
        <worldbody>
            <light pos="0 0 {max(18.0, map_spec.size):.1f}" dir="0 0 -1" directional="true"/>
            <geom name="floor" type="plane" size="{half:.3f} {half:.3f} 0.1" material="floor_mat"/>
            <geom name="wall_n" type="box" pos="0 {half:.3f} 0.6" size="{half:.3f} 0.1 0.6" rgba="0.4 0.4 0.4 0.4"/>
            <geom name="wall_s" type="box" pos="0 {-half:.3f} 0.6" size="{half:.3f} 0.1 0.6" rgba="0.4 0.4 0.4 0.4"/>
            <geom name="wall_e" type="box" pos="{half:.3f} 0 0.6" size="0.1 {half:.3f} 0.6" rgba="0.4 0.4 0.4 0.4"/>
            <geom name="wall_w" type="box" pos="{-half:.3f} 0 0.6" size="0.1 {half:.3f} 0.6" rgba="0.4 0.4 0.4 0.4"/>
            {"".join(obstacle_xml_parts)}
            {"".join(bodies)}
        </worldbody>
        <actuator>{"".join(actuators)}</actuator>
    </mujoco>"""


def render_live_assignment_overlay(viewer_handle, grid: CoverageGrid) -> None:
    """Draws translucent color-coded floor tiles showing live drone territory ownership."""
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
        self, model: mujoco.MjModel, cam_spec: CameraSpec, num_drones: int, out_dir: str
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


def save_coverage_report(
    grid: CoverageGrid, drones: List[DroneState], out_path: str
) -> None:
    """Saves a PNG showing individual and combined coverage maps + charging stations."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        parent = os.path.dirname(out_path)
        if parent:
            os.makedirs(parent, exist_ok=True)

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
            bat_lbl = f" | Bat:{d.battery:.0f}%" if grid.battery_enabled else ""
            axes[i].set_title(
                f"Drone {i}{bat_lbl}\n({grid.get_drone_coverage_pct(i):.1f}%)"
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
            )
        for d in drones:
            axes[-1].plot(d.pos[0], d.pos[1], "wo", markersize=6)
        axes[-1].set_title(f"COMBINED\n({grid.combined_coverage_pct:.1f}%)")

        plt.tight_layout()
        plt.savefig(out_path)
        plt.close(fig)
    except ImportError:
        pass


# =====================================================================
# 5. CORE SIMULATION ENGINE & EXPERIMENT BENCHMARK RUNNER
# =====================================================================

AlgoFn = Callable[[DroneState, List[DroneState], CoverageGrid], np.ndarray]


class SwarmSimulator:
    """
    Self-contained MuJoCo simulation engine.
    Instantiate with any `MapSpec` and `SimConfig`, then call `.run(algo_fn, algo_name)`.
    """

    def __init__(self, map_spec: MapSpec, config: Optional[SimConfig] = None):
        self.map_spec = map_spec
        self.config = config or SimConfig()
        self.cam_spec = CameraSpec(sensor_type=self.config.sensor_type)

        # If battery mode is enabled on a map without pads, auto-place 2 safe stations
        if self.config.battery_enabled:
            if len(map_spec.charging_stations) > 0:
                self.active_stations = np.asarray(
                    map_spec.charging_stations, dtype=np.float64
                )
            else:
                self.active_stations = place_charging_stations(
                    num_stations=2,
                    arena_size=map_spec.size,
                    obstacles=map_spec.obstacles,
                    pad_radius=map_spec.pad_radius,
                )
        else:
            self.active_stations = np.empty((0, 2), dtype=np.float64)

        xml_str = build_mujoco_xml(
            map_spec=self.map_spec,
            num_drones=self.config.num_drones,
            cam_spec=self.cam_spec,
            active_stations=self.active_stations,
            cruise_altitude=self.config.cruise_altitude,
        )
        self.model = mujoco.MjModel.from_xml_string(xml_str)
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)

        self.body_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"drone_{i}")
            for i in range(self.config.num_drones)
        ]
        self.geom_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, f"core_{i}")
            for i in range(self.config.num_drones)
        ]

        self.grid = CoverageGrid(
            num_drones=self.config.num_drones,
            size=self.map_spec.size,
            resolution=self.map_spec.resolution,
            battery_enabled=self.config.battery_enabled,
            pad_radius=self.map_spec.pad_radius,
        )
        self.grid.charging_stations = self.active_stations
        self.grid.mark_obstacles(self.model, self.data)

        comm_r = (
            self.config.comm_radius
            if self.config.comm_radius is not None
            else max(9.0, self.map_spec.size * 0.60)
        )
        self.drones = [
            DroneState(
                id=i,
                pos=self.data.qpos[i * 7 : i * 7 + 3].copy(),
                vel=np.zeros(3, dtype=np.float64),
                comm_radius=comm_r,
                target=self.data.qpos[i * 7 : i * 7 + 3].copy(),
            )
            for i in range(self.config.num_drones)
        ]

    def _update_battery_and_charging(self) -> None:
        dt = self.model.opt.timestep
        has_pads = len(self.grid.charging_stations) > 0
        scale = (
            max(1.0, self.map_spec.size / 16.0)
            if self.config.scale_battery_with_size
            else 1.0
        )

        for d in self.drones:
            if not d.active:
                d.is_charging = False
                continue

            on_pad = False
            if has_pads and d.pos[2] <= 0.45:
                pad_dists = np.linalg.norm(
                    self.grid.charging_stations - d.pos[:2], axis=1
                )
                on_pad = bool(np.min(pad_dists) <= self.grid.pad_radius)

            if on_pad and d.battery < 100.0:
                if not d.is_charging:
                    d.charge_visits += 1
                d.is_charging = True
                d.battery = min(100.0, d.battery + self.config.battery_charge_rate)
                d.charge_time += dt
            else:
                d.is_charging = False
                speed_xy = float(np.linalg.norm(d.vel[:2]))
                drain = (
                    self.config.battery_drain_base
                    + self.config.battery_drain_vel * min(1.0, speed_xy / 7.0)
                ) / scale
                d.battery = max(0.0, d.battery - drain)
                d.energy_used += drain

                if d.battery <= 0.0:
                    d.active = False
                    d.depleted = True
                    self.model.geom_rgba[self.geom_ids[d.id]] = [0.8, 0.1, 0.1, 0.5]
                    if self.config.verbose:
                        print(f"\n[BATTERY EMPTY] Drone {d.id} grounded!")

    def run(self, algo_fn: AlgoFn, algo_name: str) -> dict:
        """Executes the simulation loop and returns the full metrics dictionary."""
        cfg = self.config
        mode_tag = "with_battery" if cfg.battery_enabled else "no_battery"
        run_tag = f"{self.map_spec.name}_{mode_tag}_{algo_name}"

        slam = (
            MultiDroneSlamExtractor(
                self.model,
                self.cam_spec,
                cfg.num_drones,
                out_dir=os.path.join(cfg.output_dir, f"slam_{run_tag}"),
            )
            if cfg.save_slam_data
            else None
        )
        metrics = SwarmMetricsTracker(dt=self.model.opt.timestep)
        kp, kd, mass = 6.0, 4.5, 1.2

        def step_fn(step: int) -> bool:
            for i, d in enumerate(self.drones):
                d.pos[:] = self.data.qpos[i * 7 : i * 7 + 3]
                d.vel[:] = self.data.qvel[i * 6 : i * 6 + 3]
                self.data.qpos[i * 7 + 3 : i * 7 + 7] = (1.0, 0.0, 0.0, 0.0)
                self.data.qvel[i * 6 + 3 : i * 6 + 6] = (0.0, 0.0, 0.0)

            if (
                cfg.failure_step is not None
                and step == cfg.failure_step
                and cfg.num_drones > 1
                and self.drones[0].active
            ):
                self.drones[0].active = False
                self.model.geom_rgba[self.geom_ids[0]] = [0.8, 0.1, 0.1, 0.4]
                if cfg.verbose:
                    print(f"\n[FAILURE @ Step {step}] Drone 0 disabled!")

            if cfg.battery_enabled:
                self._update_battery_and_charging()

            if step % 4 == 0:
                self.grid.update_from_cameras(
                    self.drones, self.body_ids, self.cam_spec, self.model, self.data
                )

            if step % 10 == 0:
                all_pos = np.array([d.pos for d in self.drones])
                for i, d in enumerate(self.drones):
                    if not d.active:
                        d.target[:] = (d.pos[0], d.pos[1], 0.15)
                        continue
                    dists = np.linalg.norm(all_pos - d.pos, axis=1)
                    peers = [
                        self.drones[j]
                        for j in range(cfg.num_drones)
                        if j != i and dists[j] <= d.comm_radius
                    ]
                    tgt = algo_fn(d, peers, self.grid)
                    d.target[:] = (
                        tgt
                        if tgt is not None
                        else (d.pos[0], d.pos[1], cfg.cruise_altitude)
                    )
                    self.data.mocap_pos[i] = (d.target[0], d.target[1], 0.05)
                self.grid.update_area_assignments(self.drones, algo_name)

            if slam and step % cfg.slam_interval == 0:
                slam.capture_step(self.data, step, self.drones)

            for i, d in enumerate(self.drones):
                acc = np.clip((d.target - d.pos) * kp, -7.0, 7.0) - kd * d.vel
                self.data.ctrl[i * 3 : i * 3 + 3] = [
                    mass * acc[0],
                    mass * acc[1],
                    mass * (acc[2] + 9.81),
                ]

            mujoco.mj_step(self.model, self.data)

            cov_pct = self.grid.combined_coverage_pct
            metrics.update(step, cov_pct)

            if cfg.verbose and step % 250 == 0:
                status = " | ".join(
                    f"D{i}:{self.grid.get_drone_coverage_pct(i):.0f}%"
                    + (
                        f"(B:{self.drones[i].battery:.0f}%)"
                        if cfg.battery_enabled
                        else ""
                    )
                    for i in range(cfg.num_drones)
                )
                print(f"Step {step:04d} | Combined: {cov_pct:5.1f}% | [{status}]")

            return (cov_pct >= 99.8) or (not any(d.active for d in self.drones))

        steps_executed = 0
        if cfg.mode == "visual":
            from mujoco import viewer

            with viewer.launch_passive(self.model, self.data) as v:
                while v.is_running() and steps_executed < cfg.max_steps:
                    t0 = time.perf_counter()
                    done = step_fn(steps_executed)
                    if cfg.show_assignment:
                        render_live_assignment_overlay(v, self.grid)
                    v.sync()
                    steps_executed += 1
                    if done:
                        break
                    dt = time.perf_counter() - t0
                    if dt < self.model.opt.timestep:
                        time.sleep(self.model.opt.timestep - dt)
        else:
            for step in range(cfg.max_steps):
                steps_executed = step + 1
                if step_fn(step):
                    break

        if slam:
            slam.close()

        report = metrics.build_report(
            steps_executed,
            self.grid,
            self.drones,
            algo_name,
            self.map_spec,
            cfg.battery_enabled,
        )

        if cfg.verbose:
            SwarmMetricsTracker.print_report(report)

        if cfg.save_plots:
            os.makedirs(cfg.output_dir, exist_ok=True)
            save_coverage_report(
                self.grid,
                self.drones,
                os.path.join(cfg.output_dir, f"coverage_{run_tag}.png"),
            )

        if cfg.save_metrics:
            os.makedirs(cfg.output_dir, exist_ok=True)
            with open(
                os.path.join(cfg.output_dir, f"metrics_{run_tag}.json"),
                "w",
                encoding="utf-8",
            ) as f:
                json.dump(report, f, indent=2)

        return report


def run_simulation(
    algo_fn: AlgoFn,
    algo_name: str,
    map_spec: MapSpec,
    config: Optional[SimConfig] = None,
) -> dict:
    """Convenience wrapper to run a single simulation on `map_spec`."""
    return SwarmSimulator(map_spec=map_spec, config=config).run(algo_fn, algo_name)


def run_benchmark_suite(
    algorithms: Dict[str, AlgoFn],
    maps: Sequence[MapSpec],
    base_config: Optional[SimConfig] = None,
    scenarios: Iterable[str] = ("no_battery", "with_battery"),
    out_dir: str = "benchmark_results",
) -> List[dict]:
    """
    Tests every algorithm across all maps in both scenarios (`no_battery` and
    `with_battery`), printing a summary table and saving CSV + JSON reports.
    """
    os.makedirs(out_dir, exist_ok=True)
    template = base_config or SimConfig(
        mode="headless", save_plots=False, verbose=False
    )
    all_reports: List[dict] = []

    scenarios_list = list(scenarios)
    print(
        f"\n=== Running Benchmark Suite: {len(algorithms)} algos x "
        f"{len(maps)} maps x {len(scenarios_list)} battery scenarios ==="
    )

    for map_spec in maps:
        for scenario in scenarios_list:
            battery_on = scenario == "with_battery"
            cfg = SimConfig(
                num_drones=template.num_drones,
                max_steps=template.max_steps,
                battery_enabled=battery_on,
                failure_step=template.failure_step,
                sensor_type=template.sensor_type,
                comm_radius=template.comm_radius,
                cruise_altitude=template.cruise_altitude,
                battery_drain_base=template.battery_drain_base,
                battery_drain_vel=template.battery_drain_vel,
                battery_charge_rate=template.battery_charge_rate,
                scale_battery_with_size=template.scale_battery_with_size,
                mode="headless",
                save_plots=template.save_plots,
                save_metrics=False,
                output_dir=out_dir,
                verbose=False,
            )
            for algo_name, algo_fn in algorithms.items():
                report = run_simulation(algo_fn, algo_name, map_spec, cfg)
                all_reports.append(report)
                cov = report["coverage_time_metrics"]
                bat = report["battery_management_metrics"]
                t95 = (
                    cov["time_to_95_pct"]["time_sec"] if cov["time_to_95_pct"] else "-"
                )
                print(
                    f"[{map_spec.name:<10} | {scenario:<12} | {algo_name:<15}] "
                    f"Cov: {cov['final_coverage_pct']:5.1f}% | T95: {str(t95):>6}s | "
                    f"Energy: {bat['total_energy_drained_pct']:6.1f}% | "
                    f"Docks: {bat['total_charging_dockings']:2d} | Crashes: {bat['battery_depletion_failures']}"
                )

    # Save JSON report
    json_path = os.path.join(out_dir, "benchmark_summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_reports, f, indent=2)

    # Save flat CSV table for easy plotting / Excel / pandas analysis
    csv_path = os.path.join(out_dir, "benchmark_summary.csv")
    fieldnames = [
        "map_name",
        "scenario_mode",
        "algorithm",
        "grid_size_m",
        "grid_resolution",
        "num_obstacles",
        "final_coverage_pct",
        "total_sim_time_sec",
        "time_to_50_s",
        "time_to_80_s",
        "time_to_90_s",
        "time_to_95_s",
        "time_to_99_s",
        "total_energy_drained_pct",
        "avg_final_battery_pct",
        "min_final_battery_pct",
        "total_charging_dockings",
        "total_charging_time_sec",
        "battery_depletion_failures",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in all_reports:
            cov = r["coverage_time_metrics"]
            bat = r["battery_management_metrics"]
            writer.writerow(
                {
                    "map_name": r["map_name"],
                    "scenario_mode": r["scenario_mode"],
                    "algorithm": r["algorithm"],
                    "grid_size_m": r["grid_size_m"],
                    "grid_resolution": r["grid_resolution"],
                    "num_obstacles": r["num_obstacles"],
                    "final_coverage_pct": cov["final_coverage_pct"],
                    "total_sim_time_sec": cov["total_sim_time_sec"],
                    "time_to_50_s": (
                        cov["time_to_50_pct"]["time_sec"]
                        if cov["time_to_50_pct"]
                        else ""
                    ),
                    "time_to_80_s": (
                        cov["time_to_80_pct"]["time_sec"]
                        if cov["time_to_80_pct"]
                        else ""
                    ),
                    "time_to_90_s": (
                        cov["time_to_90_pct"]["time_sec"]
                        if cov["time_to_90_pct"]
                        else ""
                    ),
                    "time_to_95_s": (
                        cov["time_to_95_pct"]["time_sec"]
                        if cov["time_to_95_pct"]
                        else ""
                    ),
                    "time_to_99_s": (
                        cov["time_to_99_pct"]["time_sec"]
                        if cov["time_to_99_pct"]
                        else ""
                    ),
                    "total_energy_drained_pct": bat["total_energy_drained_pct"],
                    "avg_final_battery_pct": bat["avg_final_battery_pct"],
                    "min_final_battery_pct": bat["min_final_battery_pct"],
                    "total_charging_dockings": bat["total_charging_dockings"],
                    "total_charging_time_sec": bat["total_charging_time_sec"],
                    "battery_depletion_failures": bat["battery_depletion_failures"],
                }
            )

    print(f"-> Saved benchmark reports to '{csv_path}' and '{json_path}'")
    return all_reports

"""
sim_core.py — Standalone MuJoCo 3D Drone Swarm Simulation Core (Complex Shapes)
===============================================================================
Ready-to-use physics, sensor, battery, complex-shape map/dataset, A* routing,
and benchmarking engine. Keep your research focused on high-level planning in
`algorithms.py` while treating this file as a stable, decoupled testbed.

Supported Arena Shapes:
  - Convex  : "square", "circle", "ellipse"
  - Concave : "u_shape", "l_shape", "ring" (courtyard hole), "cross"
  - Custom  : Arbitrary 2D outer polygon + optional inner hole polygons in JSON

Wall Modes:
  - "soft" (1.2 m walls < 2.4 m cruise altitude): Geofenced coverage footprint;
    drones may transit over boundary cutouts.
  - "hard" (3.5 m walls > 2.4 m cruise altitude): Strict no-fly boundaries with
    built-in 2D grid A* corner routing in the flight controller so drones glide
    smoothly around concave walls (U-shape, L-shape, ring, cross) without
    modifying your high-level algorithms.
"""

from __future__ import annotations

import csv
import heapq
import json
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import imageio.v3 as iio
import mujoco
import numpy as np


# =====================================================================
# 1. POLYGON GEOMETRY ENGINE (COMPLEX & CONCAVE ARENA SHAPES)
# =====================================================================

AVAILABLE_SHAPES: Tuple[str, ...] = (
    "square",
    "circle",
    "ellipse",
    "u_shape",
    "l_shape",
    "ring",
    "cross",
)


def make_shape_polygons(
    shape: str, size: float = 16.0, **kwargs
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    Generates `(outer_polygon, hole_polygons)` as `(V, 2)` float64 vertex arrays
    centered at `(0, 0)` within `[-size/2, +size/2]`.
    """
    h = size * 0.5

    if shape == "square":
        outer = np.array([[-h, -h], [h, -h], [h, h], [-h, h]], dtype=np.float64)
        return outer, []

    if shape == "circle":
        n = int(kwargs.get("segments", 48))
        ang = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
        outer = np.column_stack([h * np.cos(ang), h * np.sin(ang)])
        return outer, []

    if shape == "ellipse":
        n = int(kwargs.get("segments", 48))
        aspect = float(kwargs.get("aspect", 0.62))
        ang = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
        outer = np.column_stack([h * np.cos(ang), (h * aspect) * np.sin(ang)])
        return outer, []

    if shape == "u_shape":
        # Concave U opening toward +Y (two vertical arms joined by a south bridge)
        arm_w = float(kwargs.get("arm_ratio", 0.34)) * size
        base_h = float(kwargs.get("base_ratio", 0.34)) * size
        ix = h - arm_w
        iy = -h + base_h
        outer = np.array(
            [
                [-h, -h],
                [h, -h],
                [h, h],
                [ix, h],
                [ix, iy],
                [-ix, iy],
                [-ix, h],
                [-h, h],
            ],
            dtype=np.float64,
        )
        return outer, []

    if shape == "l_shape":
        # Concave L missing the north-east quadrant
        arm_w = float(kwargs.get("arm_ratio", 0.44)) * size
        cx = -h + arm_w
        cy = -h + arm_w
        outer = np.array(
            [[-h, -h], [h, -h], [h, cy], [cx, cy], [cx, h], [-h, h]],
            dtype=np.float64,
        )
        return outer, []

    if shape == "ring":
        # Circular arena with a circular central courtyard hole
        n = int(kwargs.get("segments", 48))
        r_in = h * float(kwargs.get("inner_ratio", 0.42))
        ang = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
        outer = np.column_stack([h * np.cos(ang), h * np.sin(ang)])
        inner = np.column_stack([r_in * np.cos(ang), r_in * np.sin(ang)])
        return outer, [inner]

    if shape == "cross":
        # Four-wing symmetric cross
        w = float(kwargs.get("arm_ratio", 0.38)) * h
        outer = np.array(
            [
                [-w, -h],
                [w, -h],
                [w, -w],
                [h, -w],
                [h, w],
                [w, w],
                [w, h],
                [-w, h],
                [-w, w],
                [-h, w],
                [-h, -w],
                [-w, -w],
            ],
            dtype=np.float64,
        )
        return outer, []

    raise ValueError(
        f"Unknown arena shape '{shape}'. Choose from: {', '.join(AVAILABLE_SHAPES)}."
    )


def point_in_polygon(pts: np.ndarray, poly: np.ndarray) -> np.ndarray:
    """Vectorized ray-casting check for `(M, 2)` points against a `(V, 2)` polygon."""
    pts_2d = np.atleast_2d(pts)
    x = pts_2d[:, 0:1]
    y = pts_2d[:, 1:2]
    x1 = poly[:, 0][None, :]
    y1 = poly[:, 1][None, :]
    nxt = np.roll(poly, -1, axis=0)
    x2 = nxt[:, 0][None, :]
    y2 = nxt[:, 1][None, :]

    cond = (y1 > y) != (y2 > y)
    x_hit = (x2 - x1) * (y - y1) / ((y2 - y1) + 1e-12) + x1
    return (np.sum(cond & (x < x_hit), axis=1) % 2) == 1


def dist_to_polygon_edges(pts: np.ndarray, poly: np.ndarray) -> np.ndarray:
    """Minimum Euclidean distance from each of `(M, 2)` points to all edges of `poly`."""
    pts_2d = np.atleast_2d(pts)
    a = poly[None, :, :]
    b = np.roll(poly, -1, axis=0)[None, :, :]
    ab = b - a
    ab_len_sq = np.sum(ab * ab, axis=2) + 1e-12
    t = np.clip(np.sum((pts_2d[:, None, :] - a) * ab, axis=2) / ab_len_sq, 0.0, 1.0)
    proj = a + t[:, :, None] * ab
    return np.min(np.linalg.norm(pts_2d[:, None, :] - proj, axis=2), axis=1)


def points_in_arena(
    pts: np.ndarray,
    outer: np.ndarray,
    holes: Sequence[np.ndarray] = (),
    margin: float = 0.0,
) -> np.ndarray:
    """True for points inside `outer`, outside all `holes`, and `>= margin` from any wall."""
    pts_2d = np.atleast_2d(pts)
    inside = point_in_polygon(pts_2d, outer)
    for hole in holes:
        if len(hole) >= 3:
            inside &= ~point_in_polygon(pts_2d, hole)

    if margin > 0.0 and np.any(inside):
        wall_dist = dist_to_polygon_edges(pts_2d, outer)
        for hole in holes:
            if len(hole) >= 3:
                wall_dist = np.minimum(wall_dist, dist_to_polygon_edges(pts_2d, hole))
        inside &= wall_dist >= margin

    return inside


def polygon_to_walls_xml(
    poly: np.ndarray,
    wall_height: float = 1.2,
    thickness: float = 0.10,
    prefix: str = "wall",
    rgba: str = "0.40 0.42 0.48 0.55",
) -> str:
    """Converts a closed 2D polygon loop into oriented MuJoCo box wall geoms."""
    geoms = []
    n = len(poly)
    half_h = wall_height * 0.5
    for k in range(n):
        p1 = poly[k]
        p2 = poly[(k + 1) % n]
        mid = 0.5 * (p1 + p2)
        diff = p2 - p1
        length = float(np.linalg.norm(diff))
        if length < 1e-4:
            continue
        ang_deg = float(np.degrees(np.arctan2(diff[1], diff[0])))
        geoms.append(
            f'<geom name="{prefix}_{k}" type="box" '
            f'pos="{mid[0]:.3f} {mid[1]:.3f} {half_h:.3f}" '
            f'size="{length * 0.5 + thickness * 0.5:.3f} {thickness:.3f} {half_h:.3f}" '
            f'euler="0 0 {ang_deg:.2f}" rgba="{rgba}"/>'
        )
    return "\n".join(geoms)


def _farthest_point_sample(
    candidates: np.ndarray, k: int, anchor: Optional[np.ndarray] = None
) -> np.ndarray:
    """Greedily picks `k` maximally separated 2D points from `candidates`."""
    if k <= 0 or len(candidates) == 0:
        return np.empty((0, 2), dtype=np.float64)
    if k >= len(candidates):
        return candidates[:k].copy()

    if anchor is None:
        first_idx = int(np.argmax(np.linalg.norm(candidates, axis=1)))
    else:
        first_idx = int(np.argmin(np.linalg.norm(candidates - anchor, axis=1)))

    selected = [candidates[first_idx]]
    min_dists = np.linalg.norm(candidates - selected[0], axis=1)

    for _ in range(1, k):
        next_idx = int(np.argmax(min_dists))
        selected.append(candidates[next_idx])
        min_dists = np.minimum(
            min_dists, np.linalg.norm(candidates - selected[-1], axis=1)
        )

    return np.asarray(selected, dtype=np.float64)


# =====================================================================
# 2. MAP & SCENARIO SPECIFICATION (PRESETS, RANDOM MAPS, DATASETS)
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
    Supports square, circle, ellipse, U-shape, L-shape, ring, cross, or custom polygons.
    """

    name: str = "flat"
    size: float = 16.0  # Bounding-box side length in meters
    resolution: int = 40  # Grid cells per axis
    shape: str = "square"  # Shape preset or "custom"
    wall_mode: str = "soft"  # "soft" (1.2m geofence) or "hard" (3.5m no-fly walls)
    outer_polygon: List[List[float]] = field(default_factory=list)
    hole_polygons: List[List[List[float]]] = field(default_factory=list)
    obstacles: List[ObstacleSpec] = field(default_factory=list)
    charging_stations: List[List[float]] = field(default_factory=list)
    mesh_file: Optional[str] = None
    spawn_points: Optional[List[List[float]]] = None
    pad_radius: float = 0.85

    def __post_init__(self):
        if not self.outer_polygon:
            outer, holes = make_shape_polygons(self.shape, size=self.size)
            self.outer_polygon = outer.tolist()
            if not self.hole_polygons:
                self.hole_polygons = [h.tolist() for h in holes]
        if self.wall_mode not in ("soft", "hard"):
            raise ValueError(f"wall_mode must be 'soft' or 'hard', got '{self.wall_mode}'.")

    @property
    def outer_poly_np(self) -> np.ndarray:
        return np.asarray(self.outer_polygon, dtype=np.float64)

    @property
    def hole_polys_np(self) -> List[np.ndarray]:
        return [np.asarray(h, dtype=np.float64) for h in self.hole_polygons if len(h) >= 3]

    @property
    def wall_height(self) -> float:
        return 3.5 if self.wall_mode == "hard" else 1.2

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "size": float(self.size),
            "resolution": int(self.resolution),
            "shape": self.shape,
            "wall_mode": self.wall_mode,
            "outer_polygon": [[float(p[0]), float(p[1])] for p in self.outer_polygon],
            "hole_polygons": [
                [[float(p[0]), float(p[1])] for p in hole]
                for hole in self.hole_polygons
            ],
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
            shape=str(data.get("shape", "square")),
            wall_mode=str(data.get("wall_mode", "soft")),
            outer_polygon=data.get("outer_polygon", []),
            hole_polygons=data.get("hole_polygons", []),
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
        shape: str = "square",
        wall_mode: str = "soft",
        mesh_file: Optional[str] = None,
    ) -> MapSpec:
        """Creates a built-in obstacle preset inside any supported arena `shape`."""
        outer, holes = make_shape_polygons(shape, size=size)
        scale = size / 16.0
        raw_obstacles: List[ObstacleSpec] = []

        if preset == "urban":
            raw_obstacles = [
                ObstacleSpec("box", [-3.5 * scale, 3.0 * scale, 1.4], [1.4 * scale, 1.4 * scale, 1.4], name="bldg_1"),
                ObstacleSpec("box", [3.5 * scale, -3.0 * scale, 1.6], [1.5 * scale, 1.2 * scale, 1.6], name="bldg_2"),
                ObstacleSpec("box", [3.2 * scale, 3.5 * scale, 1.2], [1.2 * scale, 1.2 * scale, 1.2], rgba=[0.42, 0.35, 0.35, 1.0], name="bldg_3"),
                ObstacleSpec("box", [-3.8 * scale, -3.5 * scale, 1.5], [1.1 * scale, 1.6 * scale, 1.5], rgba=[0.35, 0.42, 0.40, 1.0], name="bldg_4"),
                ObstacleSpec("cylinder", [0.0, 0.0, 1.1], [0.9 * scale, 1.1], rgba=[0.45, 0.45, 0.50, 1.0], name="monument"),
            ]
        elif preset == "ruins":
            raw_obstacles = [
                ObstacleSpec("box", [-2.5 * scale, 0.0, 0.9], [0.3 * scale, 3.2 * scale, 0.9], rgba=[0.50, 0.42, 0.35, 1.0], name="wall_1"),
                ObstacleSpec("box", [2.5 * scale, 1.5 * scale, 0.8], [2.2 * scale, 0.3 * scale, 0.8], rgba=[0.50, 0.42, 0.35, 1.0], name="wall_2"),
                ObstacleSpec("box", [1.0 * scale, -3.0 * scale, 0.6], [1.5 * scale, 1.0 * scale, 0.6], euler=[15.0, 20.0, 35.0], rgba=[0.45, 0.40, 0.35, 1.0], name="rubble_1"),
                ObstacleSpec("cylinder", [-4.5 * scale, 4.0 * scale, 1.5], [0.6 * scale, 1.5], rgba=[0.55, 0.50, 0.45, 1.0], name="pillar_1"),
                ObstacleSpec("cylinder", [4.5 * scale, -4.0 * scale, 1.3], [0.7 * scale, 1.3], rgba=[0.55, 0.50, 0.45, 1.0], name="pillar_2"),
            ]
        elif preset == "corridors":
            raw_obstacles = [
                ObstacleSpec("box", [0.0, 3.6 * scale, 1.2], [1.3 * scale, 1.0 * scale, 1.2], name="obs_north"),
                ObstacleSpec("box", [0.0, -3.6 * scale, 1.2], [1.3 * scale, 1.0 * scale, 1.2], name="obs_south"),
            ]
        elif preset == "custom_mesh":
            if not mesh_file or not os.path.exists(mesh_file):
                raise FileNotFoundError(f"Custom mesh file not found: '{mesh_file}'")
        elif preset != "flat":
            raise ValueError(
                f"Unknown preset '{preset}'. Choose from: flat, urban, ruins, corridors, custom_mesh."
            )

        # Keep only preset obstacles that sit cleanly inside the chosen arena shape
        obstacles = [
            o for o in raw_obstacles
            if bool(points_in_arena(np.array([o.pos[:2]]), outer, holes, margin=o.footprint_radius() + 0.5)[0])
        ]

        stations = place_charging_stations(
            num_stations=num_stations,
            arena_size=size,
            outer_poly=outer,
            hole_polys=holes,
            obstacles=obstacles,
        )
        map_name = preset if shape == "square" else f"{preset}_{shape}"
        return cls(
            name=map_name,
            size=size,
            resolution=resolution,
            shape=shape,
            wall_mode=wall_mode,
            outer_polygon=outer.tolist(),
            hole_polygons=[h.tolist() for h in holes],
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
        shape: str = "square",
        wall_mode: str = "soft",
        seed: Optional[int] = None,
        name: Optional[str] = None,
    ) -> MapSpec:
        """
        Procedurally generates a random map of any shape (`square`, `circle`,
        `ellipse`, `u_shape`, `l_shape`, `ring`, `cross`, or `"random"`).
        Ensures obstacles and charging pads stay strictly inside the shape walls.
        """
        rng = np.random.default_rng(seed)
        chosen_shape = (
            str(rng.choice(AVAILABLE_SHAPES)) if shape == "random" else shape
        )
        outer, holes = make_shape_polygons(chosen_shape, size=size)
        pad_radius = 0.85

        # 1. Place charging stations first via farthest-point sampling inside shape
        stations = place_charging_stations(
            num_stations=num_stations,
            arena_size=size,
            outer_poly=outer,
            hole_polys=holes,
            obstacles=[],
            pad_radius=pad_radius,
        )

        # 2. Sample non-overlapping obstacles strictly inside the shape polygon
        obstacles: List[ObstacleSpec] = []
        half = size * 0.5
        scale = size / 16.0

        for obs_idx in range(num_obstacles):
            for _attempt in range(100):
                obs_shape = "box" if rng.random() < 0.7 else "cylinder"
                height = float(rng.uniform(0.9, 1.8))
                if obs_shape == "box":
                    hx = float(rng.uniform(0.55, 1.35) * scale)
                    hy = float(rng.uniform(0.55, 1.35) * scale)
                    obs_size = [hx, hy, height]
                    bound_r = float(np.hypot(hx, hy))
                else:
                    rad = float(rng.uniform(0.45, 1.05) * scale)
                    obs_size = [rad, height]
                    bound_r = rad

                cx = float(rng.uniform(-half + bound_r, half - bound_r))
                cy = float(rng.uniform(-half + bound_r, half - bound_r))

                # Must sit inside arena polygon with clearance from outer & hole walls
                wall_clearance = bound_r + max(0.85, 0.9 * scale)
                if not bool(points_in_arena(np.array([[cx, cy]]), outer, holes, margin=wall_clearance)[0]):
                    continue

                # Must stay clear of charging pads
                if len(stations) > 0:
                    pad_dists = np.linalg.norm(stations - np.array([cx, cy]), axis=1)
                    if np.min(pad_dists) < (bound_r + pad_radius + 1.0):
                        continue

                # Must not overlap existing obstacles (leave fly-through corridors)
                if any(
                    np.hypot(cx - ex.pos[0], cy - ex.pos[1]) < (bound_r + ex.footprint_radius() + 1.1)
                    for ex in obstacles
                ):
                    continue

                tone = float(rng.uniform(0.32, 0.52))
                obstacles.append(
                    ObstacleSpec(
                        shape=obs_shape,
                        pos=[cx, cy, height],
                        size=obs_size,
                        rgba=[tone, tone + 0.03, tone + 0.07, 1.0],
                        name=f"rand_obs_{obs_idx}",
                    )
                )
                break

        map_name = name or f"{chosen_shape}_s{seed if seed is not None else 'none'}"
        return cls(
            name=map_name,
            size=size,
            resolution=resolution,
            shape=chosen_shape,
            wall_mode=wall_mode,
            outer_polygon=outer.tolist(),
            hole_polygons=[h.tolist() for h in holes],
            obstacles=obstacles,
            charging_stations=stations.tolist(),
            pad_radius=pad_radius,
        )

    @classmethod
    def generate_dataset(
        cls,
        out_dir: str,
        num_maps: int = 7,
        size: float = 20.0,
        resolution: int = 40,
        num_obstacles_range: Tuple[int, int] = (2, 6),
        num_stations: int = 2,
        shapes: Sequence[str] = AVAILABLE_SHAPES,
        wall_mode: str = "soft",
        base_seed: int = 0,
    ) -> List[MapSpec]:
        """
        Generates `num_maps` random maps cycling through `shapes` (defaults to all
        7 shapes: square, circle, ellipse, u_shape, l_shape, ring, cross), saves
        them as JSON files in `out_dir`, and returns the list of `MapSpec`s.
        """
        os.makedirs(out_dir, exist_ok=True)
        rng = np.random.default_rng(base_seed)
        shape_list = list(shapes) if shapes else list(AVAILABLE_SHAPES)
        low_obs, high_obs = num_obstacles_range
        maps: List[MapSpec] = []

        for idx in range(num_maps):
            seed = base_seed + idx
            chosen_shape = shape_list[idx % len(shape_list)]
            n_obs = int(rng.integers(low_obs, high_obs + 1))
            m = cls.generate_random(
                size=size,
                resolution=resolution,
                num_obstacles=n_obs,
                num_stations=num_stations,
                shape=chosen_shape,
                wall_mode=wall_mode,
                seed=seed,
                name=f"map_{idx:03d}_{chosen_shape}",
            )
            m.save_json(os.path.join(out_dir, f"map_{idx:03d}_{chosen_shape}.json"))
            maps.append(m)

        return maps

    @classmethod
    def load_dataset(cls, path: str) -> List[MapSpec]:
        """Loads a directory of `.json` maps or a single JSON map/list file."""
        if os.path.isdir(path):
            files = sorted(
                os.path.join(path, fn)
                for fn in os.listdir(path)
                if fn.lower().endswith(".json")
            )
            if not files:
                raise FileNotFoundError(f"No .json map files found in directory '{path}'.")
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
    outer_poly: Optional[np.ndarray] = None,
    hole_polys: Sequence[np.ndarray] = (),
    obstacles: Sequence[ObstacleSpec] = (),
    pad_radius: float = 0.85,
) -> np.ndarray:
    """
    Places `num_stations` charging pads inside ANY polygon shape using
    farthest-point sampling over valid interior ground points.
    """
    if num_stations <= 0:
        return np.empty((0, 2), dtype=np.float64)

    if outer_poly is None or len(outer_poly) < 3:
        outer_poly, hole_polys = make_shape_polygons("square", size=arena_size)

    half = arena_size * 0.5
    samples_per_axis = 36
    coords = np.linspace(-half + 0.5, half - 0.5, samples_per_axis)
    gx, gy = np.meshgrid(coords, coords)
    grid_pts = np.stack([gx.ravel(), gy.ravel()], axis=-1)

    # Filter points inside arena with comfortable pad margin from walls and obstacles
    margin = pad_radius + 0.45
    valid = points_in_arena(grid_pts, outer_poly, hole_polys, margin=margin)
    if not np.any(valid):
        valid = points_in_arena(grid_pts, outer_poly, hole_polys, margin=0.3)

    valid_pts = grid_pts[valid]
    for obs in obstacles:
        dists = np.linalg.norm(valid_pts - np.array(obs.pos[:2]), axis=1)
        keep = dists >= (obs.footprint_radius() + pad_radius + 0.45)
        if np.any(keep):
            valid_pts = valid_pts[keep]

    if len(valid_pts) == 0:
        return np.empty((0, 2), dtype=np.float64)

    # Pull pads slightly inward from extreme corners so drones approach from all sides
    if num_stations == 1:
        centroid = np.mean(valid_pts, axis=0)
        return valid_pts[int(np.argmin(np.linalg.norm(valid_pts - centroid, axis=1))) :][:1]

    # Prefer points near 60% of arena reach along the east-west axis when available
    anchor = np.array([-arena_size * 0.30, 0.0])
    return _farthest_point_sample(valid_pts, num_stations, anchor=anchor)


# =====================================================================
# 3. SIMULATION CONFIGURATION & RUNTIME STATE STRUCTURES
# =====================================================================


@dataclass
class SimConfig:
    """Decoupled runtime settings controlling any simulation execution."""

    num_drones: int = 4
    max_steps: int = 2500
    battery_enabled: bool = False  # Toggle battery drain & charging scenario
    failure_step: Optional[int] = None  # Optional forced failure step (e.g. 500)
    sensor_type: str = "stereo"  # "mono" or "stereo"
    comm_radius: Optional[float] = None  # None -> auto-scales to max(9.0, 0.6 * map.size)
    cruise_altitude: float = 2.4
    enable_astar_routing: Optional[bool] = None  # None -> auto-ON for hard walls, OFF for soft
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

    sensor_type: str = "stereo"
    fovy: float = 65.0
    width: int = 320
    height: int = 240
    baseline: float = 0.12
    max_depth: float = 5.5
    min_depth: float = 0.4

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
    battery: float = 100.0
    is_charging: bool = False
    energy_used: float = 0.0
    charge_time: float = 0.0
    charge_visits: int = 0
    depleted: bool = False


@dataclass
class CoverageGrid:
    """Discretized ground map supporting arbitrary convex/concave polygon arenas."""

    num_drones: int
    size: float = 16.0
    resolution: int = 40
    battery_enabled: bool = False
    pad_radius: float = 0.85
    outer_polygon: Optional[np.ndarray] = None
    hole_polygons: List[np.ndarray] = field(default_factory=list)

    def __post_init__(self):
        half = self.size * 0.5
        self.cell_size = self.size / self.resolution
        step = self.cell_size * 0.5
        coords = np.linspace(-half + step, half - step, self.resolution)
        gx, gy = np.meshgrid(coords, coords)

        self.cell_centers = np.stack([gx.ravel(), gy.ravel()], axis=-1).astype(np.float64)
        self.num_cells = len(self.cell_centers)
        self.ground_points_3d = np.column_stack(
            [self.cell_centers, np.full(self.num_cells, 0.02, dtype=np.float64)]
        )

        if self.outer_polygon is None or len(self.outer_polygon) < 3:
            self.outer_polygon, self.hole_polygons = make_shape_polygons("square", size=self.size)

        # Mask of cells strictly inside the arena polygon (and outside any holes)
        self.in_arena_cells = points_in_arena(
            self.cell_centers, self.outer_polygon, self.hole_polygons, margin=0.0
        )

        self.drone_sole_visited = np.zeros((self.num_drones, self.num_cells), dtype=bool)
        self.drone_belief_uncertainty = np.ones((self.num_drones, self.num_cells), dtype=np.float32)
        self.cell_owner = np.full(self.num_cells, -1, dtype=np.int32)

        # Initialize exterior/hole cells as non-free with 0 uncertainty
        self.free_cells = self.in_arena_cells.copy()
        self.drone_belief_uncertainty[:, ~self.free_cells] = 0.0
        self.obstacle_centers = np.empty((0, 2), dtype=np.float64)
        self.total_free_cells = max(1, int(np.sum(self.free_cells)))
        self.charging_stations = np.empty((0, 2), dtype=np.float64)

    def mark_obstacles(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        """Detects static worldbody obstacles inside the arena via downward rays."""
        origin = np.array([0.0, 0.0, 10.0], dtype=np.float64)
        direction = np.array([0.0, 0.0, -1.0], dtype=np.float64)
        geom_out = np.zeros(1, dtype=np.int32)

        interior_indices = np.flatnonzero(self.in_arena_cells)
        for idx in interior_indices:
            origin[0], origin[1] = self.cell_centers[idx]
            hit = mujoco.mj_ray(model, data, origin, direction, None, 1, -1, geom_out)
            geom_id = int(geom_out[0])
            hit_body = int(model.geom_bodyid[geom_id]) if geom_id >= 0 else -1
            if 0.0 < hit < 9.75 and hit_body == 0:
                self.free_cells[idx] = False
                self.drone_belief_uncertainty[:, idx] = 0.0

        # Track only interior obstacle centers (not exterior void) for local repulsion
        self.obstacle_centers = self.cell_centers[self.in_arena_cells & (~self.free_cells)]
        self.total_free_cells = max(1, int(np.sum(self.free_cells)))

    def xy_to_rc(self, xy: np.ndarray) -> Tuple[int, int]:
        """Maps world `[x, y]` to grid `(row, col)` indices."""
        half = self.size * 0.5
        col = int(np.clip(np.floor((xy[0] + half) / self.cell_size), 0, self.resolution - 1))
        row = int(np.clip(np.floor((xy[1] + half) / self.cell_size), 0, self.resolution - 1))
        return row, col

    def rc_to_xy(self, row: int, col: int) -> np.ndarray:
        """Maps grid `(row, col)` back to world `[x, y]` cell center."""
        return self.cell_centers[row * self.resolution + col]

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
                    model, data, origin, dirs.ravel(), None, 1, body_ids[d.id],
                    geoms, hits, None, n_rays, cam_spec.max_depth + 1.0,
                )
                clear = (hits < 0.0) | (hits >= (lengths - 0.15))
            else:
                clear = np.array([
                    (hd := mujoco.mj_ray(model, data, origin, dirs[k], None, 1, body_ids[d.id], geom_out)) < 0.0
                    or hd >= (lengths[k] - 0.15)
                    for k in range(n_rays)
                ])

            visible = cand_idx[clear]
            if len(visible) > 0:
                self.drone_sole_visited[d.id, visible] = True
                self.drone_belief_uncertainty[d.id, visible] = np.maximum(
                    0.05, self.drone_belief_uncertainty[d.id, visible] * (1.0 - decay_rate)
                )

        # Peer-to-peer CRDT uncertainty map merge within comm_radius
        active_ids = [d.id for d in drones if d.active]
        if len(active_ids) > 1:
            positions = np.array([drones[i].pos for i in active_ids])
            for idx_a, i in enumerate(active_ids):
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
    """
    Pushes a 2D waypoint away from building footprints and ensures the final
    point stays strictly inside any convex or concave arena polygon.
    """
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

    # Shape-aware boundary check: snap to nearest valid free cell if pushed outside polygon
    wall_margin = min(0.55, grid.cell_size * 0.85)
    is_inside = bool(
        points_in_arena(
            np.array([target_xy]), grid.outer_polygon, grid.hole_polygons, margin=wall_margin
        )[0]
    )
    if not is_inside:
        free_pts = grid.cell_centers[grid.free_cells]
        if len(free_pts) > 0:
            # Prefer free cells that also satisfy wall margin when possible
            safe_mask = points_in_arena(
                free_pts, grid.outer_polygon, grid.hole_polygons, margin=wall_margin
            )
            pool = free_pts[safe_mask] if np.any(safe_mask) else free_pts
            target_xy = pool[int(np.argmin(np.linalg.norm(pool - target_xy, axis=1)))]

    return target_xy


# =====================================================================
# 4. GRID A* CONCAVE CORNER ROUTER (FOR HARD NO-FLY WALLS)
# =====================================================================


class GridAStarRouter:
    """
    Lightweight 8-connected grid A* router used automatically when `wall_mode="hard"`.
    Computes an intermediate line-of-sight corner waypoint whenever the straight
    path from `start_xy` to `goal_xy` crosses a concave wall cutout or obstacle.
    """

    _NEIGHBORS = (
        (-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
        (-1, -1, 1.4142), (-1, 1, 1.4142), (1, -1, 1.4142), (1, 1, 1.4142),
    )

    def __init__(self, grid: CoverageGrid, wall_clearance: float = 0.55):
        self.grid = grid
        self.res = grid.resolution
        raw_walkable = grid.free_cells.reshape(self.res, self.res).copy()
        # Keep A* corners comfortably clear of physical drone chassis + wall thickness
        safe_mask = points_in_arena(
            grid.cell_centers,
            grid.outer_polygon,
            grid.hole_polygons,
            margin=wall_clearance,
        ).reshape(self.res, self.res)
        eroded = raw_walkable & safe_mask
        self.walkable = eroded if np.any(eroded) else raw_walkable

    def _nearest_walkable(self, rc: Tuple[int, int]) -> Tuple[int, int]:
        r, c = rc
        if self.walkable[r, c]:
            return rc
        free_rc = np.argwhere(self.walkable)
        if len(free_rc) == 0:
            return rc
        dists = (free_rc[:, 0] - r) ** 2 + (free_rc[:, 1] - c) ** 2
        best = free_rc[int(np.argmin(dists))]
        return int(best[0]), int(best[1])

    def line_of_sight(self, p1: np.ndarray, p2: np.ndarray) -> bool:
        """Returns True if the straight segment between `p1` and `p2` stays on walkable cells."""
        dist = float(np.linalg.norm(p2 - p1))
        n_samples = max(2, int(np.ceil(dist / (self.grid.cell_size * 0.4))))
        ts = np.linspace(0.0, 1.0, n_samples)[:, None]
        pts = p1[None, :] + ts * (p2 - p1)[None, :]
        half = self.grid.size * 0.5
        cols = np.clip(np.floor((pts[:, 0] + half) / self.grid.cell_size).astype(int), 0, self.res - 1)
        rows = np.clip(np.floor((pts[:, 1] + half) / self.grid.cell_size).astype(int), 0, self.res - 1)
        return bool(np.all(self.walkable[rows, cols]))

    def next_waypoint(self, start_xy: np.ndarray, goal_xy: np.ndarray) -> np.ndarray:
        """Returns `goal_xy` directly if unobstructed, else the farthest visible A* corner."""
        if self.line_of_sight(start_xy, goal_xy):
            return goal_xy

        start = self._nearest_walkable(self.grid.xy_to_rc(start_xy))
        goal = self._nearest_walkable(self.grid.xy_to_rc(goal_xy))
        if start == goal:
            return goal_xy

        open_heap: List[Tuple[float, Tuple[int, int]]] = [(0.0, start)]
        came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
        g_score: Dict[Tuple[int, int], float] = {start: 0.0}

        while open_heap:
            _, curr = heapq.heappop(open_heap)
            if curr == goal:
                break
            cr, cc = curr
            curr_g = g_score[curr]
            for dr, dc, cost in self._NEIGHBORS:
                nr, nc = cr + dr, cc + dc
                if 0 <= nr < self.res and 0 <= nc < self.res and self.walkable[nr, nc]:
                    # Prevent diagonal corner-cutting through walls
                    if dr != 0 and dc != 0 and not (self.walkable[cr + dr, cc] and self.walkable[cr, cc + dc]):
                        continue
                    nxt = (nr, nc)
                    tentative_g = curr_g + cost
                    if tentative_g < g_score.get(nxt, 1e18):
                        came_from[nxt] = curr
                        g_score[nxt] = tentative_g
                        h = float(np.hypot(goal[0] - nr, goal[1] - nc))
                        heapq.heappush(open_heap, (tentative_g + h, nxt))

        if goal not in came_from:
            return goal_xy

        # Reconstruct path from start -> goal
        path_rc = [goal]
        cur = goal
        while cur in came_from:
            cur = came_from[cur]
            path_rc.append(cur)
        path_rc.reverse()

        # Pick the farthest path node visible from start_xy (smooth string-pulling lookahead)
        for rc in reversed(path_rc[1:]):
            cand_xy = self.grid.rc_to_xy(rc[0], rc[1])
            if self.line_of_sight(start_xy, cand_xy):
                return cand_xy

        return self.grid.rc_to_xy(path_rc[1][0], path_rc[1][1])


# =====================================================================
# 5. PERFORMANCE METRICS TRACKER (COVERAGE TIME & BATTERY MANAGEMENT)
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
            "shape": map_spec.shape,
            "wall_mode": map_spec.wall_mode,
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
        print("\n" + "=" * 70)
        print(
            f"METRICS | Algo: {report['algorithm'].upper()} | Map: {report['map_name']} "
            f"| Shape: {report['shape'].upper()} ({report['wall_mode']}) | Mode: {report['scenario_mode'].upper()}"
        )
        print(
            f"   Arena: {report['grid_size_m']:.1f}m ({report['grid_resolution']}x{report['grid_resolution']}) "
            f"| Obstacles: {report['num_obstacles']} | Charging pads: {report['num_charging_stations']}"
        )
        print("-" * 70)
        print("1. COVERAGE TIME:")
        for pct in (50, 80, 90, 95, 99):
            info = cov[f"time_to_{pct}_pct"]
            val = f"{info['time_sec']}s (step {info['step']})" if info else "Not reached"
            print(f"   - Time to {pct:3d}% coverage : {val}")
        print(
            f"   - Final coverage        : {cov['final_coverage_pct']:.1f}% "
            f"in {cov['total_sim_time_sec']:.2f}s ({cov['total_steps_run']} steps)"
        )
        print("2. BATTERY MANAGEMENT:")
        if report["battery_enabled"]:
            print(f"   - Total energy drained  : {bat['total_energy_drained_pct']:.1f}% (across swarm)")
            print(f"   - Final battery (avg/min): {bat['avg_final_battery_pct']:.1f}% / {bat['min_final_battery_pct']:.1f}%")
            print(f"   - Charging dockings     : {bat['total_charging_dockings']} visits ({bat['total_charging_time_sec']:.2f}s total)")
            print(f"   - Empty-battery crashes : {bat['battery_depletion_failures']} | Surviving: {bat['surviving_drones']}")
        else:
            print("   - Battery simulation disabled (no_battery scenario)")
        print("=" * 70)


# =====================================================================
# 6. XML SCENE COMPILER, VISUAL OVERLAY & SLAM EXTRACTOR
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
    """
    Returns `num_drones` valid 2D spawn points inside any convex/concave arena
    polygon, spread evenly via farthest-point sampling and clear of obstacles.
    """
    if map_spec.spawn_points and len(map_spec.spawn_points) >= num_drones:
        return [(float(p[0]), float(p[1])) for p in map_spec.spawn_points[:num_drones]]

    outer = map_spec.outer_poly_np
    holes = map_spec.hole_polys_np
    half = map_spec.size * 0.5
    coords = np.linspace(-half + 0.6, half - 0.6, 36)
    gx, gy = np.meshgrid(coords, coords)
    pts = np.stack([gx.ravel(), gy.ravel()], axis=-1)

    valid = points_in_arena(pts, outer, holes, margin=0.9)
    if not np.any(valid):
        valid = points_in_arena(pts, outer, holes, margin=0.35)
    pool = pts[valid]

    for obs in map_spec.obstacles:
        dists = np.linalg.norm(pool - np.array(obs.pos[:2]), axis=1)
        keep = dists >= (obs.footprint_radius() + 0.75)
        if np.any(keep):
            pool = pool[keep]

    chosen = _farthest_point_sample(pool, num_drones)
    return [(float(p[0]), float(p[1])) for p in chosen]


def build_mujoco_xml(
    map_spec: MapSpec,
    num_drones: int,
    cam_spec: CameraSpec,
    active_stations: np.ndarray,
    cruise_altitude: float = 2.4,
) -> str:
    """Compiles a `MapSpec` (any polygon shape + holes + obstacles) into MuJoCo XML."""
    half = map_spec.size * 0.5
    assets_xml = ""

    # 1. Outer perimeter walls + inner hole walls (soft = 1.2m, hard = 3.5m)
    wall_rgba = "0.55 0.28 0.28 0.72" if map_spec.wall_mode == "hard" else "0.40 0.42 0.48 0.50"
    geom_parts = [
        polygon_to_walls_xml(
            map_spec.outer_poly_np,
            wall_height=map_spec.wall_height,
            prefix="wall_outer",
            rgba=wall_rgba,
        )
    ]
    for hi, hole in enumerate(map_spec.hole_polys_np):
        geom_parts.append(
            polygon_to_walls_xml(
                hole,
                wall_height=map_spec.wall_height,
                prefix=f"wall_hole_{hi}",
                rgba=wall_rgba,
            )
        )

    # 2. Static 3D interior obstacles
    for i, obs in enumerate(map_spec.obstacles):
        geom_parts.append(obs.to_xml(f"obs_{i}"))

    if map_spec.mesh_file and os.path.exists(map_spec.mesh_file):
        assets_xml = f'<mesh name="custom_env_mesh" file="{map_spec.mesh_file}" scale="1 1 1"/>'
        geom_parts.append(
            '<geom name="custom_3d_map" type="mesh" mesh="custom_env_mesh" pos="0 0 0" rgba="0.5 0.5 0.55 1"/>'
        )

    # 3. Visual-only glowing green charging pads
    for i, (px, py) in enumerate(active_stations):
        geom_parts.append(
            f'<geom name="pad_{i}_outer" type="cylinder" pos="{px:.3f} {py:.3f} 0.01" '
            f'size="{map_spec.pad_radius:.2f} 0.01" rgba="0.10 0.95 0.45 0.85" contype="0" conaffinity="0"/>'
            f'<geom name="pad_{i}_inner" type="cylinder" pos="{px:.3f} {py:.3f} 0.015" '
            f'size="{map_spec.pad_radius * 0.5:.2f} 0.01" rgba="0.90 1.00 0.95 0.95" contype="0" conaffinity="0"/>'
        )

    # 4. Drones + mocap target indicators
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
            cams = f'<camera name="cam_mono_{i}" pos="0 0 -0.06" fovy="{cam_spec.fovy}"/>'
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

        actuators.append(f"""
        <general name="fx_{i}" joint="joint_{i}" gear="1 0 0 0 0 0" ctrlrange="-25 25"/>
        <general name="fy_{i}" joint="joint_{i}" gear="0 1 0 0 0 0" ctrlrange="-25 25"/>
        <general name="fz_{i}" joint="joint_{i}" gear="0 0 1 0 0 0" ctrlrange="-25 35"/>""")

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
            <geom name="floor" type="plane" size="{half + 1.0:.3f} {half + 1.0:.3f} 0.1" material="floor_mat"/>
            {"".join(geom_parts)}
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
            scn.geoms[scn.ngeom], mujoco.mjtGeom.mjGEOM_BOX, tile_size, pos, eye,
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
        self.rgb_renderer = mujoco.Renderer(model, height=cam_spec.height, width=cam_spec.width)
        self.depth_renderer = mujoco.Renderer(model, height=cam_spec.height, width=cam_spec.width)
        self.depth_renderer.enable_depth_rendering()

        fy = (cam_spec.height * 0.5) / np.tan(np.deg2rad(cam_spec.fovy * 0.5))
        self.K = np.array([[fy, 0.0, cam_spec.width * 0.5], [0.0, fy, cam_spec.height * 0.5], [0.0, 0.0, 1.0]])
        self._mj_to_cv = np.diag([1.0, -1.0, -1.0])

        os.makedirs(out_dir, exist_ok=True)
        for i in range(num_drones):
            d_dir = os.path.join(out_dir, f"drone_{i}")
            os.makedirs(os.path.join(d_dir, "rgb"), exist_ok=True)
            os.makedirs(os.path.join(d_dir, "depth"), exist_ok=True)
            with open(os.path.join(d_dir, "intrinsics.json"), "w", encoding="utf-8") as f:
                json.dump({"width": cam_spec.width, "height": cam_spec.height, "K": self.K.tolist()}, f, indent=2)

    def capture_step(self, data: mujoco.MjData, step_idx: int, drones: List[DroneState]) -> None:
        ts = step_idx * self.model.opt.timestep
        for d in drones:
            if not d.active:
                continue
            cam_name = f"cam_mono_{d.id}" if self.cam_spec.sensor_type == "mono" else f"cam_left_{d.id}"
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
            iio.imwrite(os.path.join(d_dir, "depth", f"{tag}.png"), np.clip(depth_m * 1000.0, 0, 65535).astype(np.uint16))
            with open(os.path.join(d_dir, "poses_se3.txt"), "a", encoding="utf-8") as f:
                f.write(f"{ts:.4f} {tag} " + " ".join(f"{v:.6f}" for v in T_wc.ravel()) + "\n")

    def close(self) -> None:
        self.rgb_renderer.close()
        self.depth_renderer.close()


def save_coverage_report(grid: CoverageGrid, drones: List[DroneState], out_path: str) -> None:
    """Saves a PNG showing individual and combined coverage maps + arena boundary outline."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        parent = os.path.dirname(out_path)
        if parent:
            os.makedirs(parent, exist_ok=True)

        res, half = grid.resolution, grid.size * 0.5
        outside_2d = (~grid.in_arena_cells).reshape(res, res)
        obs_2d = (grid.in_arena_cells & (~grid.free_cells)).reshape(res, res)

        def draw_outline(ax):
            for poly in [grid.outer_polygon] + list(grid.hole_polygons):
                closed = np.vstack([poly, poly[0]])
                ax.plot(closed[:, 0], closed[:, 1], "w-", linewidth=1.2, alpha=0.85)

        fig, axes = plt.subplots(1, len(drones) + 1, figsize=(3.6 * (len(drones) + 1), 3.8), dpi=140)
        for i, d in enumerate(drones):
            cov = grid.drone_sole_visited[i].reshape(res, res).astype(float)
            cov[obs_2d] = -0.35
            cov[outside_2d] = np.nan
            axes[i].set_facecolor("#181a1f")
            axes[i].imshow(cov, cmap="viridis", vmin=-0.5, vmax=1.0, origin="lower", extent=[-half, half, -half, half])
            draw_outline(axes[i])
            if len(grid.charging_stations) > 0:
                axes[i].plot(grid.charging_stations[:, 0], grid.charging_stations[:, 1], "gs", markersize=8)
            axes[i].plot(d.pos[0], d.pos[1], "ro", markersize=7)
            bat_lbl = f" | Bat:{d.battery:.0f}%" if grid.battery_enabled else ""
            axes[i].set_title(f"Drone {i}{bat_lbl}\n({grid.get_drone_coverage_pct(i):.1f}%)")

        comb = grid.combined_visited_map.reshape(res, res).astype(float)
        comb[obs_2d] = -0.35
        comb[outside_2d] = np.nan
        axes[-1].set_facecolor("#181a1f")
        axes[-1].imshow(comb, cmap="plasma", vmin=-0.5, vmax=1.0, origin="lower", extent=[-half, half, -half, half])
        draw_outline(axes[-1])
        if len(grid.charging_stations) > 0:
            axes[-1].plot(grid.charging_stations[:, 0], grid.charging_stations[:, 1], "gs", markersize=9)
        for d in drones:
            axes[-1].plot(d.pos[0], d.pos[1], "wo", markersize=6)
        axes[-1].set_title(f"COMBINED\n({grid.combined_coverage_pct:.1f}%)")

        plt.tight_layout()
        plt.savefig(out_path)
        plt.close(fig)
    except ImportError:
        pass


# =====================================================================
# 7. CORE SIMULATION ENGINE & EXPERIMENT BENCHMARK RUNNER
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

        if self.config.battery_enabled:
            if len(map_spec.charging_stations) > 0:
                self.active_stations = np.asarray(map_spec.charging_stations, dtype=np.float64)
            else:
                self.active_stations = place_charging_stations(
                    num_stations=2,
                    arena_size=map_spec.size,
                    outer_poly=map_spec.outer_poly_np,
                    hole_polys=map_spec.hole_polys_np,
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
            outer_polygon=self.map_spec.outer_poly_np,
            hole_polygons=self.map_spec.hole_polys_np,
        )
        self.grid.charging_stations = self.active_stations
        self.grid.mark_obstacles(self.model, self.data)

        # Enable A* concave corner routing automatically for hard walls (or if forced)
        use_astar = (
            self.config.enable_astar_routing
            if self.config.enable_astar_routing is not None
            else (self.map_spec.wall_mode == "hard")
        )
        self.router = GridAStarRouter(self.grid) if use_astar else None

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
        # Intermediate PD navigation targets (differ from `d.target` only when A* detours)
        self._nav_targets = [d.pos.copy() for d in self.drones]

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
                pad_dists = np.linalg.norm(self.grid.charging_stations - d.pos[:2], axis=1)
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
                self.model, self.cam_spec, cfg.num_drones,
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
                        self._nav_targets[i][:] = d.target
                        continue
                    dists = np.linalg.norm(all_pos - d.pos, axis=1)
                    peers = [
                        self.drones[j]
                        for j in range(cfg.num_drones)
                        if j != i and dists[j] <= d.comm_radius
                    ]
                    tgt = algo_fn(d, peers, self.grid)
                    d.target[:] = (
                        tgt if tgt is not None else (d.pos[0], d.pos[1], cfg.cruise_altitude)
                    )
                    self.data.mocap_pos[i] = (d.target[0], d.target[1], 0.05)

                    # Route around concave hard walls automatically without altering `d.target`
                    if self.router is not None:
                        wp_xy = self.router.next_waypoint(d.pos[:2], d.target[:2])
                        self._nav_targets[i][:] = (wp_xy[0], wp_xy[1], d.target[2])
                    else:
                        self._nav_targets[i][:] = d.target

                self.grid.update_area_assignments(self.drones, algo_name)

            if slam and step % cfg.slam_interval == 0:
                slam.capture_step(self.data, step, self.drones)

            for i, d in enumerate(self.drones):
                acc = np.clip((self._nav_targets[i] - d.pos) * kp, -7.0, 7.0) - kd * d.vel
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
                    + (f"(B:{self.drones[i].battery:.0f}%)" if cfg.battery_enabled else "")
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
            steps_executed, self.grid, self.drones, algo_name, self.map_spec, cfg.battery_enabled
        )

        if cfg.verbose:
            SwarmMetricsTracker.print_report(report)

        if cfg.save_plots:
            os.makedirs(cfg.output_dir, exist_ok=True)
            save_coverage_report(
                self.grid, self.drones, os.path.join(cfg.output_dir, f"coverage_{run_tag}.png")
            )

        if cfg.save_metrics:
            os.makedirs(cfg.output_dir, exist_ok=True)
            with open(os.path.join(cfg.output_dir, f"metrics_{run_tag}.json"), "w", encoding="utf-8") as f:
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
    template = base_config or SimConfig(mode="headless", save_plots=False, verbose=False)
    scenarios_list = list(scenarios)
    all_reports: List[dict] = []

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
                enable_astar_routing=template.enable_astar_routing,
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
                t95 = cov["time_to_95_pct"]["time_sec"] if cov["time_to_95_pct"] else "-"
                print(
                    f"[{map_spec.name:<16} | {map_spec.shape:<7} | {scenario:<12} | {algo_name:<15}] "
                    f"Cov: {cov['final_coverage_pct']:5.1f}% | T95: {str(t95):>6}s | "
                    f"Energy: {bat['total_energy_drained_pct']:6.1f}% | "
                    f"Docks: {bat['total_charging_dockings']:2d} | Crashes: {bat['battery_depletion_failures']}"
                )

    json_path = os.path.join(out_dir, "benchmark_summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_reports, f, indent=2)

    csv_path = os.path.join(out_dir, "benchmark_summary.csv")
    fieldnames = [
        "map_name", "shape", "wall_mode", "scenario_mode", "algorithm",
        "grid_size_m", "grid_resolution", "num_obstacles",
        "final_coverage_pct", "total_sim_time_sec",
        "time_to_50_s", "time_to_80_s", "time_to_90_s", "time_to_95_s", "time_to_99_s",
        "total_energy_drained_pct", "avg_final_battery_pct", "min_final_battery_pct",
        "total_charging_dockings", "total_charging_time_sec", "battery_depletion_failures",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in all_reports:
            cov = r["coverage_time_metrics"]
            bat = r["battery_management_metrics"]
            writer.writerow({
                "map_name": r["map_name"],
                "shape": r["shape"],
                "wall_mode": r["wall_mode"],
                "scenario_mode": r["scenario_mode"],
                "algorithm": r["algorithm"],
                "grid_size_m": r["grid_size_m"],
                "grid_resolution": r["grid_resolution"],
                "num_obstacles": r["num_obstacles"],
                "final_coverage_pct": cov["final_coverage_pct"],
                "total_sim_time_sec": cov["total_sim_time_sec"],
                "time_to_50_s": cov["time_to_50_pct"]["time_sec"] if cov["time_to_50_pct"] else "",
                "time_to_80_s": cov["time_to_80_pct"]["time_sec"] if cov["time_to_80_pct"] else "",
                "time_to_90_s": cov["time_to_90_pct"]["time_sec"] if cov["time_to_90_pct"] else "",
                "time_to_95_s": cov["time_to_95_pct"]["time_sec"] if cov["time_to_95_pct"] else "",
                "time_to_99_s": cov["time_to_99_pct"]["time_sec"] if cov["time_to_99_pct"] else "",
                "total_energy_drained_pct": bat["total_energy_drained_pct"],
                "avg_final_battery_pct": bat["avg_final_battery_pct"],
                "min_final_battery_pct": bat["min_final_battery_pct"],
                "total_charging_dockings": bat["total_charging_dockings"],
                "total_charging_time_sec": bat["total_charging_time_sec"],
                "battery_depletion_failures": bat["battery_depletion_failures"],
            })

    print(f"-> Saved benchmark reports to '{csv_path}' and '{json_path}'")
    return all_reports

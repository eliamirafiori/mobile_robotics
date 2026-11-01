import argparse
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List
import numpy as np
import mujoco

# =====================================================================
# 1. DATA STRUCTURES & DECENTRALIZED STATE
# =====================================================================


@dataclass
class DroneState:
    id: int
    pos: np.ndarray  # [x, y, z]
    vel: np.ndarray  # [vx, vy, vz]
    yaw: float  # Heading angle (rad)
    active: bool = True  # False if crashed / battery depleted
    fov_radius: float = 2.2  # Sensing footprint radius (m)
    comm_radius: float = 8.0  # Decentralized neighbor discovery radius (m)
    target: np.ndarray = field(default_factory=lambda: np.zeros(3))


@dataclass
class CoverageGrid:
    """2D discretization of the arena floor for tracking coverage & uncertainty."""

    size: float = 16.0  # Arena goes from -size/2 to +size/2
    resolution: int = 40  # 40x40 grid cells

    def __post_init__(self):
        half = self.size / 2.0
        coords = np.linspace(
            -half + (half / self.resolution),
            half - (half / self.resolution),
            self.resolution,
        )
        gx, gy = np.meshgrid(coords, coords)
        self.cell_centers = np.stack([gx.ravel(), gy.ravel()], axis=-1)  # (N_cells, 2)
        # Importance / uncertainty weight: 1.0 = unvisited, decays toward 0.05 when covered
        self.uncertainty = np.ones(len(self.cell_centers), dtype=np.float32)
        self.visited = np.zeros(len(self.cell_centers), dtype=bool)

    def update(
        self,
        drones: List[DroneState],
        decay_rate: float = 0.15,
        forget_rate: float = 0.0005,
    ):
        # Slight information decay over time (for persistent monitoring)
        self.uncertainty = np.clip(self.uncertainty + forget_rate, 0.05, 1.0)

        for d in drones:
            if not d.active:
                continue
            dists = np.linalg.norm(self.cell_centers - d.pos[:2], axis=1)
            in_fov = dists <= d.fov_radius
            self.visited[in_fov] = True
            self.uncertainty[in_fov] = np.maximum(
                0.05, self.uncertainty[in_fov] * (1.0 - decay_rate)
            )

    @property
    def coverage_ratio(self) -> float:
        return float(np.mean(self.visited)) * 100.0


# =====================================================================
# 2. SWAPPABLE COVERAGE ALGORITHM REGISTRY
# =====================================================================

AlgorithmFunc = Callable[[DroneState, List[DroneState], CoverageGrid], np.ndarray]
ALGORITHMS: Dict[str, AlgorithmFunc] = {}


def register_algorithm(name: str):
    """Decorator to register a new coverage algorithm by name."""

    def decorator(func: AlgorithmFunc):
        ALGORITHMS[name] = func
        return func

    return decorator


@register_algorithm("voronoi")
def voronoi_lloyd_coverage(
    ego: DroneState, neighbors: List[DroneState], grid: CoverageGrid
) -> np.ndarray:
    """
    Decentralized Centroidal Voronoi Tessellation (Lloyd's Algorithm).
    Each drone computes its local Voronoi cell using only active neighbors within
    communication range and moves toward the uncertainty-weighted centroid.
    """
    pts = grid.cell_centers
    weights = grid.uncertainty**2  # Prioritize highly uncertain cells

    # Compute distances from all grid cells to ego drone
    ego_dists = np.linalg.norm(pts - ego.pos[:2], axis=1)
    in_ego_cell = np.ones(len(pts), dtype=bool)

    # Filter grid cells closer to any active neighbor than to ego
    for nb in neighbors:
        if not nb.active or nb.id == ego.id:
            continue
        nb_dists = np.linalg.norm(pts - nb.pos[:2], axis=1)
        in_ego_cell &= ego_dists <= nb_dists

    if not np.any(in_ego_cell):
        return ego.pos.copy()

    cell_pts = pts[in_ego_cell]
    cell_weights = weights[in_ego_cell]
    centroid_2d = np.average(cell_pts, axis=0, weights=cell_weights)

    return np.array([centroid_2d[0], centroid_2d[1], 2.0])


@register_algorithm("frontier")
def marvel_colmar_frontier_coverage(
    ego: DroneState, neighbors: List[DroneState], grid: CoverageGrid
) -> np.ndarray:
    """
    Cooperative Frontier / Viewpoint Utility Selection (inspired by MARVEL & COLMAR).
    Samples candidate viewpoints, scores them by expected information gain within FOV,
    and penalizes overlap with neighbors' current targets and positions.
    """
    pts = grid.cell_centers
    unc = grid.uncertainty

    # Candidate viewpoints: top uncertain cells
    high_unc_idx = np.argsort(unc)[-60:]
    candidates = pts[high_unc_idx]

    best_score = -1e9
    best_target = ego.pos[:2]

    for cand in candidates:
        # 1. Information gain around candidate viewpoint
        d_cells = np.linalg.norm(pts - cand, axis=1)
        info_gain = np.sum(unc[d_cells <= ego.fov_radius])

        # 2. Travel cost from ego drone
        dist_cost = np.linalg.norm(cand - ego.pos[:2]) * 1.8

        # 3. Cooperative coordination penalty (avoid clustering around neighbor targets)
        coord_penalty = 0.0
        for nb in neighbors:
            if not nb.active or nb.id == ego.id:
                continue
            d_nb_target = np.linalg.norm(cand - nb.target[:2])
            d_nb_pos = np.linalg.norm(cand - nb.pos[:2])
            if d_nb_target < ego.fov_radius * 1.5:
                coord_penalty += 15.0 / (d_nb_target + 0.5)
            if d_nb_pos < ego.fov_radius:
                coord_penalty += 10.0

        score = info_gain - dist_cost - coord_penalty
        if score > best_score:
            best_score = score
            best_target = cand

    return np.array([best_target[0], best_target[1], 2.0])


@register_algorithm("potential_field")
def potential_field_coverage(
    ego: DroneState, neighbors: List[DroneState], grid: CoverageGrid
) -> np.ndarray:
    """
    Decentralized Artificial Potential Field (APF).
    Repelled by nearby active drones; attracted by nearby high-uncertainty cells.
    """
    force = np.zeros(2)

    # Attractive force toward uncertain cells
    vecs = grid.cell_centers - ego.pos[:2]
    dists = np.linalg.norm(vecs, axis=1) + 1e-4
    attr_weights = (grid.uncertainty**3) / (dists**1.5)
    force += np.sum((vecs / dists[:, None]) * attr_weights[:, None], axis=0) * 0.08

    # Repulsive force from neighbors
    for nb in neighbors:
        if not nb.active or nb.id == ego.id:
            continue
        diff = ego.pos[:2] - nb.pos[:2]
        d = np.linalg.norm(diff) + 1e-4
        if d < 4.5:
            force += (diff / d) * (6.0 / (d**2))

    # Step in direction of resultant force
    norm = np.linalg.norm(force)
    step = (force / norm) * min(norm, 2.5) if norm > 1e-3 else np.zeros(2)
    target_2d = np.clip(ego.pos[:2] + step, -grid.size / 2 + 1.0, grid.size / 2 - 1.0)
    return np.array([target_2d[0], target_2d[1], 2.0])


# =====================================================================
# 3. MUJOCO ARENA GENERATOR & EVENT MANAGER
# =====================================================================


def build_swarm_xml(num_drones: int, arena_size: float = 16.0) -> str:
    """Generates a MuJoCo MJCF XML model with N quadrotor bodies and visual target markers."""
    half = arena_size / 2.0
    drone_bodies = []
    actuators = []

    colors = [
        "0.2 0.8 0.3 1",
        "0.2 0.6 1.0 1",
        "1.0 0.6 0.1 1",
        "0.9 0.2 0.8 1",
        "0.1 0.9 0.9 1",
        "0.9 0.9 0.2 1",
    ]

    for i in range(num_drones):
        angle = 2 * np.pi * i / num_drones
        sx, sy = 2.0 * np.cos(angle), 2.0 * np.sin(angle)
        color = colors[i % len(colors)]

        drone_bodies.append(f"""
        <body name="drone_{i}" pos="{sx:.2f} {sy:.2f} 2.0">
            <freejoint name="joint_{i}"/>
            <geom name="core_{i}" type="box" size="0.22 0.22 0.06" rgba="{color}" mass="1.0"/>
            <geom type="cylinder" pos="0.22 0.22 0.04" size="0.12 0.01" rgba="0.8 0.8 0.8 0.7" mass="0.05"/>
            <geom type="cylinder" pos="-0.22 0.22 0.04" size="0.12 0.01" rgba="0.8 0.8 0.8 0.7" mass="0.05"/>
            <geom type="cylinder" pos="0.22 -0.22 0.04" size="0.12 0.01" rgba="0.8 0.8 0.8 0.7" mass="0.05"/>
            <geom type="cylinder" pos="-0.22 -0.22 0.04" size="0.12 0.01" rgba="0.8 0.8 0.8 0.7" mass="0.05"/>
            <!-- FOV Ground Projection Ring -->
            <site name="fov_{i}" pos="0 0 -1.95" size="2.2 2.2 0.01" type="cylinder" rgba="{color[:11]} 0.18"/>
        </body>
        <!-- Mocap marker showing each drone's current algorithm target -->
        <body name="target_vis_{i}" mocap="true" pos="{sx:.2f} {sy:.2f} 0.05">
            <geom name="target_geom_{i}" type="sphere" size="0.18" rgba="{color}" contype="0" conaffinity="0"/>
        </body>
        """)

        actuators.append(f"""
        <general name="fx_{i}" joint="joint_{i}" gear="1 0 0 0 0 0" ctrlrange="-25 25"/>
        <general name="fy_{i}" joint="joint_{i}" gear="0 1 0 0 0 0" ctrlrange="-25 25"/>
        <general name="fz_{i}" joint="joint_{i}" gear="0 0 1 0 0 0" ctrlrange="-25 35"/>
        """)

    return f"""
    <mujoco model="drone_swarm_coverage">
        <option timestep="0.01" gravity="0 0 -9.81" integrator="RK4"/>
        <visual>
            <headlight diffuse="0.7 0.7 0.7" ambient="0.3 0.3 0.3"/>
        </visual>
        <worldbody>
            <light pos="0 0 15" dir="0 0 -1" directional="true"/>
            <geom name="floor" type="plane" size="{half} {half} 0.1" rgba="0.18 0.20 0.24 1"/>
            <!-- Arena Boundary Walls -->
            <geom type="box" pos="0 {half} 0.5" size="{half} 0.1 0.5" rgba="0.4 0.4 0.4 0.5"/>
            <geom type="box" pos="0 {-half} 0.5" size="{half} 0.1 0.5" rgba="0.4 0.4 0.4 0.5"/>
            <geom type="box" pos="{half} 0 0.5" size="0.1 {half} 0.5" rgba="0.4 0.4 0.4 0.5"/>
            <geom type="box" pos="{-half} 0 0.5" size="0.1 {half} 0.5" rgba="0.4 0.4 0.4 0.5"/>
            {"".join(drone_bodies)}
        </worldbody>
        <actuator>
            {"".join(actuators)}
        </actuator>
    </mujoco>
    """


def trigger_swarm_events(step: int, drones: List[DroneState], model: mujoco.MjModel):
    """
    Injects dynamic swarm events to test decentralized self-healing:
      - Step 400: Drone 0 suffers critical failure and drops out.
      - Step 800: Drone 1 suffers partial sensor degradation (reduced FOV).
      - Step 1100: Drone 0 recovers and rejoins the swarm.
    """
    if step == 400 and len(drones) > 0:
        drones[0].active = False
        geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "core_0")
        model.geom_rgba[geom_id] = [0.8, 0.1, 0.1, 0.4]  # Turn dim red on failure
        print(
            f"\n[EVENT @ Step {step}] Drone 0 FAILED! Remaining swarm redistributing area..."
        )

    elif step == 800 and len(drones) > 1:
        drones[1].fov_radius = 1.2
        print(
            f"\n[EVENT @ Step {step}] Drone 1 FOV degraded to 1.2m! Adapting coverage..."
        )

    elif step == 1100 and len(drones) > 0:
        drones[0].active = True
        geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "core_0")
        model.geom_rgba[geom_id] = [0.2, 0.8, 0.3, 1.0]
        print(f"\n[EVENT @ Step {step}] Drone 0 RECOVERED and rejoined swarm!")


# =====================================================================
# 4. SIMULATION LOOP (HEADLESS & VISUAL MODES)
# =====================================================================


def run_simulation(algo_name: str, mode: str, num_drones: int, max_steps: int):
    algo_fn = ALGORITHMS[algo_name]
    xml_str = build_swarm_xml(num_drones=num_drones)
    model = mujoco.MjModel.from_xml_string(xml_str)
    data = mujoco.MjData(model)
    grid = CoverageGrid(size=16.0, resolution=40)

    drones = [
        DroneState(id=i, pos=np.zeros(3), vel=np.zeros(3), yaw=0.0)
        for i in range(num_drones)
    ]

    print(
        f"=== Starting Swarm Simulation | Mode: {mode.upper()} | Algorithm: {algo_name.upper()} ==="
    )

    def step_simulation(step_idx: int):
        # 1. Sync state from MuJoCo physics
        for i, d in enumerate(drones):
            qpos_adr = i * 7
            qvel_adr = i * 6
            d.pos = data.qpos[qpos_adr : qpos_adr + 3].copy()
            d.vel = data.qvel[qvel_adr : qvel_adr + 3].copy()
            # Lock orientation upright to focus on 3D position + coverage control
            data.qpos[qpos_adr + 3 : qpos_adr + 7] = [1.0, 0.0, 0.0, 0.0]
            data.qvel[qvel_adr + 3 : qvel_adr + 6] = [0.0, 0.0, 0.0]

        # 2. Check for dynamic swarm events (failures, sensor degradation, recovery)
        trigger_swarm_events(step_idx, drones, model)

        # 3. Update coverage map
        grid.update(drones)

        # 4. Recompute decentralized targets every 10 physics steps (10 Hz planner)
        if step_idx % 10 == 0:
            for d in drones:
                if not d.active:
                    d.target = np.array([d.pos[0], d.pos[1], 0.1])  # Land on failure
                    continue
                # Decentralized communication filter: only see neighbors within comm_radius
                local_neighbors = [
                    nb
                    for nb in drones
                    if nb.id != d.id and np.linalg.norm(nb.pos - d.pos) <= d.comm_radius
                ]
                d.target = algo_fn(d, local_neighbors, grid)
                data.mocap_pos[d.id] = [d.target[0], d.target[1], 0.05]

        # 5. Low-level PD Flight Controller + Gravity Compensation
        kp, kd = 6.0, 4.5
        drone_mass = 1.2  # Core + 4 rotors
        for i, d in enumerate(drones):
            pos_err = d.target - d.pos
            # Clamp max horizontal speed to ~2.5 m/s
            vel_cmd = np.clip(pos_err * kp, -8.0, 8.0)
            acc_cmd = vel_cmd - kd * d.vel
            data.ctrl[i * 3 + 0] = drone_mass * acc_cmd[0]
            data.ctrl[i * 3 + 1] = drone_mass * acc_cmd[1]
            data.ctrl[i * 3 + 2] = drone_mass * (acc_cmd[2] + 9.81)

        mujoco.mj_step(model, data)

        if step_idx % 200 == 0:
            active_cnt = sum(1 for d in drones if d.active)
            print(
                f"Step {step_idx:04d}/{max_steps} | "
                f"Active Drones: {active_cnt}/{num_drones} | "
                f"Area Visited: {grid.coverage_ratio:.1f}% | "
                f"Mean Uncertainty: {np.mean(grid.uncertainty):.3f}"
            )

    # Execute in selected mode
    if mode == "visual":
        from mujoco import viewer  # <-- Fixes UnboundLocalError by binding 'viewer', not 'mujoco'
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
        # Headless Mode (also verifies GPU EGL offscreen rendering)
        renderer = mujoco.Renderer(model, height=480, width=640)
        for step_idx in range(max_steps):
            step_simulation(step_idx)

        # Save a snapshot of the final swarm state to the shared volume
        renderer.update_scene(data)
        frame = renderer.render()
        import imageio.v3 as iio

        iio.imwrite("headless_final_snapshot.png", frame)
        renderer.close()
        print("Saved offscreen GPU render to 'headless_final_snapshot.png'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Decentralized Drone Swarm Coverage Testbed"
    )
    parser.add_argument(
        "--algo",
        type=str,
        default="voronoi",
        choices=list(ALGORITHMS.keys()),
        help="Coverage algorithm to execute",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="headless",
        choices=["headless", "visual"],
        help="Run in headless offscreen GPU mode or interactive GUI mode",
    )
    parser.add_argument(
        "--drones", type=int, default=5, help="Number of drones in swarm"
    )
    parser.add_argument("--steps", type=int, default=1600, help="Simulation steps")
    args = parser.parse_args()

    run_simulation(
        algo_name=args.algo,
        mode=args.mode,
        num_drones=args.drones,
        max_steps=args.steps,
    )

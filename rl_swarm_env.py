"""
Multi-Agent Reinforcement Learning (MARL) Environment & PPO Integration
========================================================================
Wraps the MuJoCo 3D drone swarm simulation (`swarm_coverage.py`) into a
standard Gymnasium environment for Centralized Training with Decentralized
Execution (CTDE).

Architecture Overview:
  - Shared Observation Builder (`build_drone_observation`): Guarantees 100%
    identical state representations during PPO training and deployment.
  - Decentralized Execution: Every drone runs an identical copy of the trained
    neural network using only local state, nearby neighbors, and its own
    peer-to-peer belief map.
  - Plugin Integration (`@register_algorithm("rl_ppo")`): Registers the trained
    policy directly into `swarm_coverage.py` so it can be benchmarked side-by-side
    against Voronoi, Frontier, and Spiral algorithms.
"""

import argparse
import os
from typing import List, Optional

import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np

from swarm_coverage import (
    CameraSpec,
    CoverageGrid,
    DroneState,
    apply_obstacle_clearance,
    build_swarm_xml,
    register_algorithm,
    run_simulation,
)

# =====================================================================
# 1. SHARED DECENTRALIZED OBSERVATION BUILDER (TRAIN & INFERENCE)
# =====================================================================

MAX_OBSERVED_NEIGHBORS = 3
POOLED_GRID_DIM = 10
OBS_DIM = 4 + (MAX_OBSERVED_NEIGHBORS * 2) + (POOLED_GRID_DIM * POOLED_GRID_DIM)  # 110


def build_drone_observation(
    ego: DroneState, neighbors: List[DroneState], grid: CoverageGrid
) -> np.ndarray:
    """
    Constructs a fixed-size (110-D) local observation vector for a single drone.
    Using a single shared function for both RL training and `@register_algorithm`
    inference eliminates train/test feature mismatch.

    Observation Layout:
      [0:4]     -> Normalized ego position (x, y) and velocity (vx, vy)
      [4:10]    -> Relative (dx, dy) offsets to the 3 closest active neighbors
                   within `comm_radius` (padded with 2.0 if fewer than 3 exist)
      [10:110]  -> 10x10 average-pooled local uncertainty map from ego's belief grid
    """
    half_arena = grid.size * 0.5

    # 1. Ego kinematic state normalized to roughly [-1, 1]
    ego_kinematics = np.array(
        [
            ego.pos[0] / half_arena,
            ego.pos[1] / half_arena,
            ego.vel[0] / 7.0,
            ego.vel[1] / 7.0,
        ],
        dtype=np.float32,
    )

    # 2. Nearest active neighbors within radio communication range
    visible_peers = []
    for peer in neighbors:
        if peer.id != ego.id and peer.active:
            dist_3d = float(np.linalg.norm(peer.pos - ego.pos))
            if dist_3d <= ego.comm_radius:
                rel_xy = (peer.pos[:2] - ego.pos[:2]) / half_arena
                visible_peers.append((dist_3d, rel_xy))

    visible_peers.sort(key=lambda item: item[0])

    # Pad missing neighbor slots with 2.0 (outside normalized arena bounds)
    neighbor_features = np.full((MAX_OBSERVED_NEIGHBORS, 2), 2.0, dtype=np.float32)
    for slot in range(min(MAX_OBSERVED_NEIGHBORS, len(visible_peers))):
        neighbor_features[slot] = visible_peers[slot][1]

    # 3. Downsample ego's 2D belief uncertainty map (e.g., 40x40 -> 10x10)
    res = grid.resolution
    pool_factor = res // POOLED_GRID_DIM
    unc_2d = grid.drone_belief_uncertainty[ego.id].reshape(res, res)
    pooled_belief = (
        unc_2d.reshape(POOLED_GRID_DIM, pool_factor, POOLED_GRID_DIM, pool_factor)
        .mean(axis=(1, 3))
        .ravel()
        .astype(np.float32)
    )

    return np.concatenate([ego_kinematics, neighbor_features.ravel(), pooled_belief])


# =====================================================================
# 2. GYMNASIUM MULTI-AGENT SWARM ENVIRONMENT (CTDE)
# =====================================================================


class DecentralizedSwarmEnv(gym.Env):
    """
    Gymnasium environment wrapping `swarm_coverage.py` for Multi-Agent RL.

    During training:
      - Drone 0 is controlled directly by the RL trainer (`action_drone_0`).
      - Drones 1..N-1 execute a batched forward pass of the same shared policy
        network (self-play parameter sharing).
      - Random drone failures are injected mid-episode so the learned policy
        automatically adapts and covers dropped teammates' territories.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(
        self,
        num_drones: int = 4,
        scenario: str = "flat",
        sensor_type: str = "stereo",
        max_decision_steps: int = 200,  # 200 decisions * 10 physics steps = 2000 steps (20s)
        inject_failures: bool = True,
    ):
        super().__init__()
        self.num_drones = num_drones
        self.scenario = scenario
        self.cam_spec = CameraSpec(sensor_type=sensor_type)
        self.max_decision_steps = max_decision_steps
        self.inject_failures = inject_failures

        # Build MuJoCo physics model once and cache body IDs
        xml_str = build_swarm_xml(num_drones, self.cam_spec, scenario=scenario)
        self.model = mujoco.MjModel.from_xml_string(xml_str)
        self.data = mujoco.MjData(self.model)
        self.drone_body_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"drone_{i}")
            for i in range(num_drones)
        ]

        # Precompute static 3D obstacle mask once to make episode resets instantaneous
        mujoco.mj_forward(self.model, self.data)
        template_grid = CoverageGrid(num_drones=num_drones, size=16.0, resolution=40)
        template_grid.mark_obstacles(self.model, self.data)
        self._cached_free_cells = template_grid.free_cells.copy()
        self._cached_obstacle_centers = template_grid.obstacle_centers.copy()
        self._cached_total_free_cells = template_grid.total_free_cells

        # Continuous 2D target waypoint command in normalized coordinates [-1, 1]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-2.0, high=2.0, shape=(OBS_DIM,), dtype=np.float32
        )

        # Shared policy reference updated during PPO training for teammate self-play
        self.shared_policy = None

        self.grid: Optional[CoverageGrid] = None
        self.drones: List[DroneState] = []
        self.decision_step = 0
        self.failure_step = -1

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)

        # Spawn drones at randomized perimeter angles/radii to prevent overfitting
        for i in range(self.num_drones):
            angle = (2.0 * np.pi * i / self.num_drones) + self.np_random.uniform(-0.35, 0.35)
            radius = self.np_random.uniform(3.5, 6.5)
            self.data.qpos[i * 7 : i * 7 + 3] = (
                radius * np.cos(angle),
                radius * np.sin(angle),
                2.4,
            )

        mujoco.mj_forward(self.model, self.data)

        # Initialize fresh coverage grid using cached static obstacle geometry
        self.grid = CoverageGrid(num_drones=self.num_drones, size=16.0, resolution=40)
        self.grid.free_cells[:] = self._cached_free_cells
        self.grid.obstacle_centers = self._cached_obstacle_centers
        self.grid.total_free_cells = self._cached_total_free_cells
        self.grid.drone_belief_uncertainty[:, ~self._cached_free_cells] = 0.0

        self.drones = [
            DroneState(
                id=i,
                pos=self.data.qpos[i * 7 : i * 7 + 3].copy(),
                vel=np.zeros(3, dtype=np.float64),
                target=self.data.qpos[i * 7 : i * 7 + 3].copy(),
            )
            for i in range(self.num_drones)
        ]

        self.decision_step = 0
        self.failure_step = (
            int(self.np_random.integers(40, 120)) if self.inject_failures else -1
        )

        # Perform initial camera scan before first decision
        self.grid.update_from_cameras(
            self.drones, self.drone_body_ids, self.cam_spec, self.model, self.data
        )
        return build_drone_observation(self.drones[0], self.drones, self.grid), {}

    def step(self, action_drone_0: np.ndarray):
        """
        Advances simulation by 10 physics steps (0.1s decision interval).
        Batches teammate policy inference into a single forward pass for speed.
        """
        self.decision_step += 1

        # Simulate sudden battery/motor failure on a teammate mid-episode
        if self.decision_step == self.failure_step and self.num_drones > 1:
            victim_id = self.num_drones - 1
            self.drones[victim_id].active = False

        prev_swarm_covered = int(
            np.sum(self.grid.combined_visited_map & self.grid.free_cells)
        )
        prev_ego_covered = int(
            np.sum(self.grid.drone_sole_visited[0] & self.grid.free_cells)
        )

        # 1. Batch-predict actions for all active teammates (Drones 1..N-1)
        active_teammates = [d for d in self.drones[1:] if d.active]
        teammate_actions = {}
        if active_teammates:
            if self.shared_policy is not None:
                obs_batch = np.stack(
                    [build_drone_observation(d, self.drones, self.grid) for d in active_teammates]
                )
                acts_batch, _ = self.shared_policy.predict(obs_batch, deterministic=False)
                for d, act in zip(active_teammates, acts_batch):
                    teammate_actions[d.id] = act
            else:
                for d in active_teammates:
                    teammate_actions[d.id] = self.action_space.sample()

        # 2. Convert normalized [-1, 1] network outputs into safe 3D target waypoints
        half_arena = self.grid.size * 0.5
        for i, d in enumerate(self.drones):
            if not d.active:
                d.target[:] = (d.pos[0], d.pos[1], 0.15)
                continue

            raw_act = action_drone_0 if i == 0 else teammate_actions[d.id]
            raw_xy = np.clip(raw_act, -1.0, 1.0) * (half_arena - 0.5)
            safe_xy = apply_obstacle_clearance(raw_xy, self.grid)
            d.target[:] = (safe_xy[0], safe_xy[1], 2.4)

        # 3. Step MuJoCo physics 10 times (100 Hz physics -> 10 Hz RL control)
        kp, kd, mass = 6.0, 4.5, 1.2
        gravity_force = mass * 9.81

        for sub_step in range(10):
            for i, d in enumerate(self.drones):
                qpos_adr, qvel_adr = i * 7, i * 6
                d.pos[:] = self.data.qpos[qpos_adr : qpos_adr + 3]
                d.vel[:] = self.data.qvel[qvel_adr : qvel_adr + 3]
                self.data.qpos[qpos_adr + 3 : qpos_adr + 7] = (1.0, 0.0, 0.0, 0.0)
                self.data.qvel[qvel_adr + 3 : qvel_adr + 6] = (0.0, 0.0, 0.0)

                vel_cmd = np.clip((d.target - d.pos) * kp, -7.0, 7.0)
                acc_cmd = vel_cmd - kd * d.vel
                ctrl_adr = i * 3
                self.data.ctrl[ctrl_adr] = mass * acc_cmd[0]
                self.data.ctrl[ctrl_adr + 1] = mass * acc_cmd[1]
                self.data.ctrl[ctrl_adr + 2] = mass * acc_cmd[2] + gravity_force

            mujoco.mj_step(self.model, self.data)

            if sub_step % 4 == 0:
                self.grid.update_from_cameras(
                    self.drones,
                    self.drone_body_ids,
                    self.cam_spec,
                    self.model,
                    self.data,
                )

        # 4. Compute multi-objective reward for Drone 0
        new_swarm_covered = int(
            np.sum(self.grid.combined_visited_map & self.grid.free_cells)
        )
        new_ego_covered = int(
            np.sum(self.grid.drone_sole_visited[0] & self.grid.free_cells)
        )

        ego_discovery_gain = new_ego_covered - prev_ego_covered
        swarm_discovery_gain = new_swarm_covered - prev_swarm_covered

        # Penalize flying too close (< 1.5m) to active teammates
        crowding_penalty = sum(
            0.3
            for nb in self.drones[1:]
            if nb.active and np.linalg.norm(nb.pos[:2] - self.drones[0].pos[:2]) < 1.5
        )

        reward = (
            (0.15 * ego_discovery_gain)
            + (0.05 * swarm_discovery_gain)
            - crowding_penalty
            - 0.02  # Small step cost to encourage fast completion
        )

        coverage_pct = self.grid.combined_coverage_pct
        terminated = bool(coverage_pct >= 99.0)
        truncated = bool(self.decision_step >= self.max_decision_steps)

        if terminated:
            reward += 15.0

        obs_0 = build_drone_observation(self.drones[0], self.drones, self.grid)
        return obs_0, float(reward), terminated, truncated, {"coverage": coverage_pct}


# =====================================================================
# 3. PPO TRAINING & `@register_algorithm("rl_ppo")` PLUGIN
# =====================================================================

_LOADED_RL_POLICY = None


def train_ppo(
    total_timesteps: int = 150_000,
    scenario: str = "flat",
    num_drones: int = 4,
    model_path: str = "ppo_swarm_coverage",
) -> None:
    """Trains a decentralized PPO policy and saves the weights to disk."""
    from stable_baselines3 import PPO

    env = DecentralizedSwarmEnv(
        num_drones=num_drones, scenario=scenario, inject_failures=True
    )
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=128,
        gamma=0.99,
        verbose=1,
    )
    # Link shared policy so teammates immediately use updated weights during rollouts
    env.shared_policy = model

    print(f"=== Training Decentralized PPO Policy ({total_timesteps} steps | {scenario.upper()}) ===")
    model.learn(total_timesteps=total_timesteps)
    model.save(model_path)
    print(f"-> Saved trained RL policy to '{model_path}.zip'")


@register_algorithm("rl_ppo")
def rl_ppo_coverage(
    ego: DroneState, neighbors: List[DroneState], grid: CoverageGrid
) -> np.ndarray:
    """
    Inference wrapper allowing `swarm_coverage.py` to execute the trained PPO policy
    using the exact same `--algo rl_ppo` interface as Voronoi or Frontier.
    """
    global _LOADED_RL_POLICY
    if _LOADED_RL_POLICY is None:
        from stable_baselines3 import PPO

        if not os.path.exists("ppo_swarm_coverage.zip"):
            raise FileNotFoundError(
                "Trained model 'ppo_swarm_coverage.zip' not found. "
                "Run `python rl_swarm_env.py --train` first."
            )
        _LOADED_RL_POLICY = PPO.load("ppo_swarm_coverage")

    obs = build_drone_observation(ego, neighbors, grid)
    action, _ = _LOADED_RL_POLICY.predict(obs, deterministic=True)

    half_arena = grid.size * 0.5
    raw_xy = np.clip(action, -1.0, 1.0) * (half_arena - 0.5)
    safe_xy = apply_obstacle_clearance(raw_xy, grid)
    return np.array([float(safe_xy[0]), float(safe_xy[1]), 2.4], dtype=np.float64)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or Evaluate Decentralized PPO Swarm")
    parser.add_argument("--train", action="store_true", help="Train a new PPO policy")
    parser.add_argument("--timesteps", type=int, default=150_000, help="Total PPO training steps")
    parser.add_argument(
        "--scenario", type=str, default="flat", choices=["flat", "urban", "ruins"]
    )
    parser.add_argument(
        "--mode", type=str, default="headless", choices=["visual", "headless"]
    )
    parser.add_argument("--sensor", type=str, default="stereo", choices=["mono", "stereo"])
    parser.add_argument("--drones", type=int, default=4)
    parser.add_argument("--steps", type=int, default=1500)
    args = parser.parse_args()

    if args.train:
        train_ppo(
            total_timesteps=args.timesteps,
            scenario=args.scenario,
            num_drones=args.drones,
        )
    else:
        run_simulation(
            algo_name="rl_ppo",
            mode=args.mode,  # Fixed: properly forwards CLI --mode (default: headless)
            sensor_type=args.sensor,
            scenario=args.scenario,
            custom_mesh=None,
            num_drones=args.drones,
            max_steps=args.steps,
            show_assignment=True,
            save_slam_data=False,
            slam_interval=25,
        )

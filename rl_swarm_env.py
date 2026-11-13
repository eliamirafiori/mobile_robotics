import argparse
import os
import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np

# Import your existing simulation components directly from swarm_coverage.py
from swarm_coverage import (
    CameraSpec,
    CoverageGrid,
    DroneState,
    apply_obstacle_clearance,
    build_swarm_xml,
    register_algorithm,
)


class DecentralizedSwarmEnv(gym.Env):
    """
    Gymnasium environment wrapping swarm_coverage.py for Multi-Agent RL.
    Trains a shared decentralized policy across all active drones in the swarm.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(
        self,
        num_drones: int = 4,
        scenario: str = "flat",
        sensor_type: str = "stereo",
        max_decision_steps: int = 200,  # 200 decision steps * 10 physics steps = 2000 physics steps
        inject_failures: bool = True,
    ):
        super().__init__()
        self.num_drones = num_drones
        self.scenario = scenario
        self.cam_spec = CameraSpec(sensor_type=sensor_type)
        self.max_decision_steps = max_decision_steps
        self.inject_failures = inject_failures

        # Build MuJoCo model once
        xml_str = build_swarm_xml(num_drones, self.cam_spec, scenario=scenario)
        self.model = mujoco.MjModel.from_xml_string(xml_str)
        self.data = mujoco.MjData(self.model)

        self.drone_body_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"drone_{i}")
            for i in range(num_drones)
        ]

        # Action: normalized 2D target waypoint [tx, ty] in [-1, 1]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        # Observation: 4 (ego state) + 6 (3 nearest neighbors dx, dy) + 100 (10x10 belief grid) = 110
        self.observation_space = spaces.Box(
            low=-2.0, high=2.0, shape=(110,), dtype=np.float32
        )

        self.grid: CoverageGrid | None = None
        self.drones: list[DroneState] = []
        self.decision_step = 0
        self.failure_step = -1

    def _get_drone_obs(self, ego_idx: int) -> np.ndarray:
        """Constructs the local decentralized observation vector for a single drone."""
        ego = self.drones[ego_idx]
        half_arena = self.grid.size * 0.5

        # 1. Normalized ego position & velocity (4)
        ego_state = np.array(
            [
                ego.pos[0] / half_arena,
                ego.pos[1] / half_arena,
                ego.vel[0] / 7.0,
                ego.vel[1] / 7.0,
            ],
            dtype=np.float32,
        )

        # 2. Relative positions of up to 3 closest active neighbors within comm_radius (6)
        rel_neighbors = []
        for nb in self.drones:
            if nb.id != ego.id and nb.active:
                diff = (nb.pos[:2] - ego.pos[:2]) / half_arena
                dist = np.linalg.norm(nb.pos - ego.pos)
                if dist <= ego.comm_radius:
                    rel_neighbors.append((dist, diff))

        rel_neighbors.sort(key=lambda item: item[0])
        neighbor_feats = np.full((3, 2), 2.0, dtype=np.float32)
        for k in range(min(3, len(rel_neighbors))):
            neighbor_feats[k] = rel_neighbors[k][1]

        # 3. Downsample 40x40 local uncertainty belief map to 10x10 (100)
        unc_2d = self.grid.drone_belief_uncertainty[ego.id].reshape(40, 40)
        pooled_unc = (
            unc_2d.reshape(10, 4, 10, 4).mean(axis=(1, 3)).ravel().astype(np.float32)
        )

        return np.concatenate([ego_state, neighbor_feats.ravel(), pooled_unc])

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)

        # Randomize starting positions slightly for better RL generalization
        for i in range(self.num_drones):
            angle = (2.0 * np.pi * i / self.num_drones) + self.np_random.uniform(
                -0.3, 0.3
            )
            radius = self.np_random.uniform(3.5, 6.5)
            self.data.qpos[i * 7 : i * 7 + 3] = [
                radius * np.cos(angle),
                radius * np.sin(angle),
                2.4,
            ]

        mujoco.mj_forward(self.model, self.data)

        self.grid = CoverageGrid(num_drones=self.num_drones, size=16.0, resolution=40)
        self.grid.mark_obstacles(self.model, self.data)

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
        # Randomly schedule a drone dropout during training so the policy learns self-healing
        self.failure_step = (
            int(self.np_random.integers(40, 120)) if self.inject_failures else -1
        )

        # Initial sensor scan
        self.grid.update_from_cameras(
            self.drones, self.drone_body_ids, self.cam_spec, self.model, self.data
        )
        # Return Drone 0's observation for standard single-agent SB3 wrapper
        return self._get_drone_obs(0), {}

    def step(self, action_drone_0: np.ndarray, policy_model=None):
        """
        Executes 10 physics steps (0.1s). Drone 0 uses `action_drone_0` from the RL trainer,
        while Drones 1..N query the same shared `policy_model` for decentralized execution.
        """
        self.decision_step += 1

        # Inject dynamic drone failure during training
        if self.decision_step == self.failure_step and self.num_drones > 1:
            victim_id = self.num_drones - 1
            self.drones[victim_id].active = False

        prev_global_covered = int(
            np.sum(self.grid.combined_visited_map & self.grid.free_cells)
        )
        prev_ego_covered = int(
            np.sum(self.grid.drone_sole_visited[0] & self.grid.free_cells)
        )

        # Assign targets to all active drones using the shared policy
        half_arena = self.grid.size * 0.5
        for i, d in enumerate(self.drones):
            if not d.active:
                d.target[:] = (d.pos[0], d.pos[1], 0.15)
                continue

            if i == 0:
                act = action_drone_0
            elif policy_model is not None:
                obs_i = self._get_drone_obs(i)
                act, _ = policy_model.predict(obs_i, deterministic=False)
            else:
                act = self.action_space.sample()

            raw_xy = np.clip(act, -1.0, 1.0) * (half_arena - 0.5)
            safe_xy = apply_obstacle_clearance(raw_xy, self.grid)
            d.target[:] = (safe_xy[0], safe_xy[1], 2.4)

        # Step MuJoCo physics 10 times (matching your 10 Hz planner loop)
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
                self.data.ctrl[i * 3 : i * 3 + 3] = [
                    mass * acc_cmd[0],
                    mass * acc_cmd[1],
                    mass * acc_cmd[2] + gravity_force,
                ]

            mujoco.mj_step(self.model, self.data)

            if sub_step % 4 == 0:
                self.grid.update_from_cameras(
                    self.drones,
                    self.drone_body_ids,
                    self.cam_spec,
                    self.model,
                    self.data,
                )

        # Compute reward for Drone 0
        new_global_covered = int(
            np.sum(self.grid.combined_visited_map & self.grid.free_cells)
        )
        new_ego_covered = int(
            np.sum(self.grid.drone_sole_visited[0] & self.grid.free_cells)
        )

        ego_gain = new_ego_covered - prev_ego_covered
        swarm_gain = new_global_covered - prev_global_covered

        # Proximity penalty if too close to another active drone
        crowd_penalty = 0.0
        for nb in self.drones[1:]:
            if nb.active and np.linalg.norm(nb.pos[:2] - self.drones[0].pos[:2]) < 1.5:
                crowd_penalty += 0.3

        reward = (0.15 * ego_gain) + (0.05 * swarm_gain) - crowd_penalty - 0.02

        coverage_pct = self.grid.combined_coverage_pct
        terminated = bool(coverage_pct >= 99.0)
        truncated = bool(self.decision_step >= self.max_decision_steps)

        if terminated:
            reward += 15.0  # Completion bonus

        return (
            self._get_drone_obs(0),
            float(reward),
            terminated,
            truncated,
            {"coverage": coverage_pct},
        )


# =====================================================================
# TRAINING & PLUGIN INTEGRATION
# =====================================================================


def train_ppo(total_timesteps: int = 150_000, scenario: str = "flat"):
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback

    env = DecentralizedSwarmEnv(num_drones=4, scenario=scenario, inject_failures=True)
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=128,
        gamma=0.99,
        verbose=1,
    )

    # Keep teammates updated with the latest policy weights during training (self-play)
    class SelfPlaySyncCallback(BaseCallback):
        def _on_step(self) -> bool:
            return True

    # Wrap env.step so teammates always use the current `model`
    orig_step = env.step
    env.step = lambda action: orig_step(action, policy_model=model)

    print(f"=== Training Decentralized PPO Policy ({total_timesteps} steps) ===")
    model.learn(total_timesteps=total_timesteps, callback=SelfPlaySyncCallback())
    model.save("ppo_swarm_coverage")
    print("-> Saved trained RL policy to 'ppo_swarm_coverage.zip'")


# Register the trained RL model directly into your existing @register_algorithm system!
_LOADED_RL_POLICY = None


@register_algorithm("rl_ppo")
def rl_ppo_coverage(
    ego: DroneState, neighbors: list[DroneState], grid: CoverageGrid
) -> np.ndarray:
    """Runs the trained PPO policy inside swarm_coverage.py's visual/headless runner."""
    global _LOADED_RL_POLICY
    if _LOADED_RL_POLICY is None:
        from stable_baselines3 import PPO

        if not os.path.exists("ppo_swarm_coverage.zip"):
            raise FileNotFoundError("Train first with: python rl_swarm_env.py --train")
        _LOADED_RL_POLICY = PPO.load("ppo_swarm_coverage")

    half_arena = grid.size * 0.5
    ego_state = np.array(
        [
            ego.pos[0] / half_arena,
            ego.pos[1] / half_arena,
            ego.vel[0] / 7.0,
            ego.vel[1] / 7.0,
        ],
        dtype=np.float32,
    )

    rel_neighbors = sorted(
        [
            (np.linalg.norm(nb.pos - ego.pos), (nb.pos[:2] - ego.pos[:2]) / half_arena)
            for nb in neighbors
            if nb.active and nb.id != ego.id
        ],
        key=lambda x: x[0],
    )
    neighbor_feats = np.full((3, 2), 2.0, dtype=np.float32)
    for k in range(min(3, len(rel_neighbors))):
        neighbor_feats[k] = rel_neighbors[k][1]

    unc_2d = grid.drone_belief_uncertainty[ego.id].reshape(40, 40)
    pooled_unc = (
        unc_2d.reshape(10, 4, 10, 4).mean(axis=(1, 3)).ravel().astype(np.float32)
    )

    obs = np.concatenate([ego_state, neighbor_feats.ravel(), pooled_unc])
    action, _ = _LOADED_RL_POLICY.predict(obs, deterministic=True)

    raw_xy = np.clip(action, -1.0, 1.0) * (half_arena - 0.5)
    safe_xy = apply_obstacle_clearance(raw_xy, grid)
    return np.array([float(safe_xy[0]), float(safe_xy[1]), 2.4], dtype=np.float64)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train", action="store_true", help="Train the decentralized PPO policy"
    )
    parser.add_argument("--timesteps", type=int, default=150_000)
    parser.add_argument(
        "--scenario", type=str, default="flat", choices=["flat", "urban", "ruins"]
    )
    parser.add_argument("--mode", type=str, default="headless", choices=["visual", "headless"])
    args = parser.parse_args()

    if args.train:
        train_ppo(total_timesteps=args.timesteps, scenario=args.scenario)
    else:
        # Run your main simulation with `--algo rl_ppo` registered!
        from swarm_coverage import run_simulation

        run_simulation(
            algo_name="rl_ppo",
            mode="visual",
            sensor_type="stereo",
            scenario=args.scenario,
            custom_mesh=None,
            num_drones=4,
            max_steps=1500,
            show_assignment=True,
            save_slam_data=False,
            slam_interval=25,
        )

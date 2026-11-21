"""run_swarm.py — Command-Line Entry Point for Swarm Coverage Experiments"""

import argparse
from algorithms import ALGORITHMS
from sim_core import run_simulation

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Decentralized Drone Swarm Coverage")
    parser.add_argument(
        "--algo", type=str, default="battery_patrol", choices=list(ALGORITHMS.keys())
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
        choices=["flat", "urban", "ruins", "charging", "custom_mesh"],
    )
    parser.add_argument("--mesh-file", type=str, default=None)
    parser.add_argument("--drones", type=int, default=4)
    parser.add_argument("--grid-size", type=float, default=16)
    parser.add_argument("--grid-resolution", type=int, default=40)
    parser.add_argument("--steps", type=int, default=2500)
    args = parser.parse_args()

    run_simulation(
        algo_fn=ALGORITHMS[args.algo],
        algo_name=args.algo,
        mode=args.mode,
        sensor_type=args.sensor,
        scenario=args.scenario,
        custom_mesh=args.mesh_file,
        num_drones=args.drones,
        size=args.grid_size,
        resolution=args.grid_resolution,
        max_steps=args.steps,
    )

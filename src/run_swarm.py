"""run_swarm.py — Command-Line Entry Point for Swarm Coverage Experiments"""

import argparse
from algorithms import ALGORITHMS
from sim_core import run_simulation

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Decentralized Drone Swarm Coverage")
    parser.add_argument(
        "--algo", type=str, default="voronoi", choices=list(ALGORITHMS.keys())
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
        choices=["flat", "urban", "ruins", "custom_mesh"],
    )
    parser.add_argument("--mesh-file", type=str, default=None)
    parser.add_argument("--drones", type=int, default=4)
    parser.add_argument("--steps", type=int, default=2000)
    args = parser.parse_args()

    run_simulation(
        algo_fn=ALGORITHMS[args.algo],
        algo_name=args.algo,
        mode=args.mode,
        sensor_type=args.sensor,
        scenario=args.scenario,
        custom_mesh=args.mesh_file,
        num_drones=args.drones,
        max_steps=args.steps,
    )

"""
run_swarm.py — CLI Runner for Single Simulations, Map Datasets & Benchmarks
===========================================================================
Examples:
  1. Single visual run on a preset map (with battery enabled):
       python run_swarm.py --algo battery_patrol --map urban --battery --mode visual

  2. Generate a reproducible dataset of 10 random maps:
       python run_swarm.py --generate-dataset maps_dataset --num-maps 10 --grid-size 20 --grid-res 40

  3. Benchmark selected algorithms across a dataset in BOTH scenarios (no_battery & with_battery):
       python run_swarm.py --benchmark --dataset maps_dataset --algos voronoi patrol battery_patrol battery_voronoi

  4. Benchmark directly on 5 freshly generated random maps in BOTH battery scenarios:
       python run_swarm.py --benchmark --num-maps 5 --grid-size 20 --grid-res 40 --algos patrol battery_patrol
"""

import argparse
import os
from typing import List

from algorithms import get_algorithm, list_algorithms, select_algorithms
from sim_core import MapSpec, SimConfig, run_benchmark_suite, run_simulation


def resolve_single_map(args: argparse.Namespace) -> MapSpec:
    """Loads a map from a JSON file, generates a random map, or builds a preset."""
    if os.path.isfile(args.map):
        return MapSpec.load_json(args.map)
    if args.map == "random":
        return MapSpec.generate_random(
            size=args.grid_size,
            resolution=args.grid_res,
            num_obstacles=args.obstacles,
            num_stations=args.stations,
            seed=args.seed,
        )
    return MapSpec.from_preset(
        preset=args.map,
        size=args.grid_size,
        resolution=args.grid_res,
        num_stations=args.stations,
        mesh_file=args.mesh_file,
    )


def resolve_benchmark_maps(args: argparse.Namespace) -> List[MapSpec]:
    """Loads maps from `--dataset` or generates `--num-maps` random maps."""
    if args.dataset:
        return MapSpec.load_dataset(args.dataset)
    return [
        MapSpec.generate_random(
            size=args.grid_size,
            resolution=args.grid_res,
            num_obstacles=args.obstacles,
            num_stations=args.stations,
            seed=args.seed + i,
            name=f"rand_{i:03d}",
        )
        for i in range(args.num_maps)
    ]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Decentralized Drone Swarm Testbed")

    # Execution mode flags
    parser.add_argument(
        "--generate-dataset",
        type=str,
        default=None,
        help="Generate N random JSON maps into this directory and exit",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run all selected algorithms across maps in both no_battery & with_battery scenarios",
    )

    # Algorithm selection
    parser.add_argument(
        "--algo",
        type=str,
        default="battery_patrol",
        choices=list_algorithms(),
        help="Algorithm for single-run mode",
    )
    parser.add_argument(
        "--algos",
        nargs="+",
        default=None,
        choices=list_algorithms(),
        help="List of algorithms for --benchmark mode (default: all coverage algorithms)",
    )

    # Map / Dataset specification
    parser.add_argument(
        "--map",
        type=str,
        default="urban",
        help="Preset name (flat, urban, ruins, corridors, custom_mesh), 'random', or path to a .json map",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Directory of .json maps (or single dataset .json) for --benchmark",
    )
    parser.add_argument(
        "--num-maps", type=int, default=5, help="Number of random maps to generate"
    )
    parser.add_argument(
        "--obstacles", type=int, default=5, help="Obstacle count for random maps"
    )
    parser.add_argument(
        "--stations", type=int, default=2, help="Charging station count"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Base random seed for map generation"
    )
    parser.add_argument(
        "--mesh-file", type=str, default=None, help="Optional .stl/.obj mesh"
    )

    # Simulation parameters
    parser.add_argument(
        "--mode", type=str, default="headless", choices=["headless", "visual"]
    )
    parser.add_argument(
        "--sensor", type=str, default="stereo", choices=["mono", "stereo"]
    )
    parser.add_argument("--drones", type=int, default=4)
    parser.add_argument("--grid-size", type=float, default=16.0)
    parser.add_argument("--grid-res", type=int, default=40)
    parser.add_argument("--steps", type=int, default=2500)
    parser.add_argument(
        "--battery",
        action="store_true",
        help="Enable battery drain and charging stations (single-run mode)",
    )
    parser.add_argument(
        "--failure-step",
        type=int,
        default=None,
        help="Optional step at which Drone 0 fails (default: disabled)",
    )
    parser.add_argument(
        "--out-dir", type=str, default="results", help="Output folder for reports/plots"
    )

    args = parser.parse_args()

    # 1. Generate a reusable random map dataset
    if args.generate_dataset:
        maps = MapSpec.generate_dataset(
            out_dir=args.generate_dataset,
            num_maps=args.num_maps,
            size=args.grid_size,
            resolution=args.grid_res,
            num_obstacles_range=(max(1, args.obstacles - 2), args.obstacles + 2),
            num_stations=args.stations,
            base_seed=args.seed,
        )
        print(f"-> Generated {len(maps)} random maps in '{args.generate_dataset}/'")

    # 2. Benchmark suite: every selected algorithm x every map x (no_battery, with_battery)
    elif args.benchmark:
        maps = resolve_benchmark_maps(args)
        algos = select_algorithms(args.algos)
        base_cfg = SimConfig(
            num_drones=args.drones,
            max_steps=args.steps,
            failure_step=args.failure_step,
            sensor_type=args.sensor,
            mode="headless",
            save_plots=False,
            output_dir=args.out_dir,
        )
        run_benchmark_suite(
            algorithms=algos,
            maps=maps,
            base_config=base_cfg,
            scenarios=("no_battery", "with_battery"),
            out_dir=args.out_dir,
        )

    # 3. Single simulation run
    else:
        map_spec = resolve_single_map(args)
        cfg = SimConfig(
            num_drones=args.drones,
            max_steps=args.steps,
            battery_enabled=args.battery,
            failure_step=args.failure_step,
            sensor_type=args.sensor,
            mode=args.mode,
            output_dir=args.out_dir,
        )
        run_simulation(
            algo_fn=get_algorithm(args.algo),
            algo_name=args.algo,
            map_spec=map_spec,
            config=cfg,
        )

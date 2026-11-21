"""
run_swarm.py — CLI Runner for Complex-Shape Simulations, Datasets & Benchmarks
==============================================================================
Examples:
  1. Single visual run on a concave U-shape with hard walls + A* corner routing:
       python run_swarm.py --algo battery_patrol --map flat --shape u_shape --wall-mode hard --battery --mode visual

  2. Generate a mixed-shape dataset of 7 random maps (square, circle, ellipse, U, L, ring, cross):
       python run_swarm.py --generate-dataset complex_maps --num-maps 7 --grid-size 20 --grid-res 40 --wall-mode hard

  3. Benchmark algorithms across the mixed-shape dataset in BOTH battery scenarios:
       python run_swarm.py --benchmark --dataset complex_maps --algos voronoi patrol battery_patrol battery_voronoi --steps 3500
"""

import argparse
import os
from typing import List

from algorithms import get_algorithm, list_algorithms, select_algorithms
from sim_core import (
    AVAILABLE_SHAPES,
    MapSpec,
    SimConfig,
    run_benchmark_suite,
    run_simulation,
)


def resolve_single_map(args: argparse.Namespace) -> MapSpec:
    """Loads a map JSON file, generates a random shaped map, or builds a shaped preset."""
    if os.path.isfile(args.map):
        return MapSpec.load_json(args.map)
    if args.map == "random":
        return MapSpec.generate_random(
            size=args.grid_size,
            resolution=args.grid_res,
            num_obstacles=args.obstacles,
            num_stations=args.stations,
            shape=args.shape,
            wall_mode=args.wall_mode,
            seed=args.seed,
        )
    chosen_shape = "square" if args.shape == "random" else args.shape
    return MapSpec.from_preset(
        preset=args.map,
        size=args.grid_size,
        resolution=args.grid_res,
        num_stations=args.stations,
        shape=chosen_shape,
        wall_mode=args.wall_mode,
        mesh_file=args.mesh_file,
    )


def resolve_benchmark_maps(args: argparse.Namespace) -> List[MapSpec]:
    """Loads maps from `--dataset` or generates `--num-maps` random maps across `--shapes`."""
    if args.dataset:
        return MapSpec.load_dataset(args.dataset)
    shape_pool = args.shapes if args.shapes else list(AVAILABLE_SHAPES)
    return [
        MapSpec.generate_random(
            size=args.grid_size,
            resolution=args.grid_res,
            num_obstacles=args.obstacles,
            num_stations=args.stations,
            shape=shape_pool[i % len(shape_pool)],
            wall_mode=args.wall_mode,
            seed=args.seed + i,
            name=f"rand_{i:03d}_{shape_pool[i % len(shape_pool)]}",
        )
        for i in range(args.num_maps)
    ]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Decentralized Drone Swarm Testbed (Complex Shapes)")

    # Execution mode flags
    parser.add_argument(
        "--generate-dataset",
        type=str,
        default=None,
        help="Generate N random complex-shape JSON maps into this folder and exit",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run all selected algorithms across maps in both no_battery & with_battery scenarios",
    )

    # Algorithm selection
    parser.add_argument("--algo", type=str, default="battery_patrol", choices=list_algorithms())
    parser.add_argument("--algos", nargs="+", default=None, choices=list_algorithms())

    # Map, Shape & Dataset specification
    parser.add_argument(
        "--map",
        type=str,
        default="flat",
        help="Preset (flat, urban, ruins, corridors, custom_mesh), 'random', or path to a .json map",
    )
    parser.add_argument(
        "--shape",
        type=str,
        default="square",
        choices=list(AVAILABLE_SHAPES) + ["random"],
        help="Arena boundary shape for single-run mode",
    )
    parser.add_argument(
        "--shapes",
        nargs="+",
        default=None,
        choices=list(AVAILABLE_SHAPES),
        help="Shapes to cycle through for --generate-dataset or --benchmark (default: all 7 shapes)",
    )
    parser.add_argument(
        "--wall-mode",
        type=str,
        default="soft",
        choices=["soft", "hard"],
        help="'soft' = 1.2m geofence walls; 'hard' = 3.5m no-fly walls with automatic A* corner routing",
    )
    parser.add_argument("--dataset", type=str, default=None, help="Directory or JSON file of maps")
    parser.add_argument("--num-maps", type=int, default=7, help="Number of random maps to generate")
    parser.add_argument("--obstacles", type=int, default=4, help="Obstacle count for random maps")
    parser.add_argument("--stations", type=int, default=2, help="Charging station count")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed")
    parser.add_argument("--mesh-file", type=str, default=None, help="Optional .stl/.obj mesh")

    # Simulation parameters
    parser.add_argument("--mode", type=str, default="headless", choices=["headless", "visual"])
    parser.add_argument("--sensor", type=str, default="stereo", choices=["mono", "stereo"])
    parser.add_argument("--drones", type=int, default=4)
    parser.add_argument("--grid-size", type=float, default=20.0)
    parser.add_argument("--grid-res", type=int, default=40)
    parser.add_argument("--steps", type=int, default=2500)
    parser.add_argument("--battery", action="store_true", help="Enable battery & charging (single run)")
    parser.add_argument("--failure-step", type=int, default=None, help="Optional step to disable Drone 0")
    parser.add_argument("--out-dir", type=str, default="results", help="Output directory")

    args = parser.parse_args()

    if args.generate_dataset:
        shape_pool = args.shapes if args.shapes else list(AVAILABLE_SHAPES)
        maps = MapSpec.generate_dataset(
            out_dir=args.generate_dataset,
            num_maps=args.num_maps,
            size=args.grid_size,
            resolution=args.grid_res,
            num_obstacles_range=(max(0, args.obstacles - 2), args.obstacles + 2),
            num_stations=args.stations,
            shapes=shape_pool,
            wall_mode=args.wall_mode,
            base_seed=args.seed,
        )
        print(f"-> Generated {len(maps)} maps ({', '.join(shape_pool)}) in '{args.generate_dataset}/'")

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

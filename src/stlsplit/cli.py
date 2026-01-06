from __future__ import annotations

import argparse
import logging
from dataclasses import replace
from pathlib import Path
from typing import Optional

from .config import AppConfig, ConnectorConfig, OutputConfig, SplitConfig, BuildVolume, load_config
from .pipeline import run_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Split 3D meshes into printer-sized pieces.")
    parser.add_argument("input", type=Path, help="Input mesh (STL/OBJ/3MF)")
    parser.add_argument("--output", type=Path, help="Output path (default: output/<name>.3mf)")
    parser.add_argument("--config", type=Path, help="Path to config.yaml")
    parser.add_argument("--format", choices=["3mf", "stl"], help="Output format")
    parser.add_argument("--engine", help="Boolean engine (manifold/scad/blender)")
    parser.add_argument("--build", nargs=3, type=float, metavar=("X", "Y", "Z"), help="Build volume in mm")
    parser.add_argument("--connector", choices=["auto", "hex", "dovetail", "magnet", "none"], help="Connector style")
    parser.add_argument("--tolerance", type=float, help="Connector tolerance in mm")
    parser.add_argument("--peg-radius", type=float, help="Connector peg radius in mm")
    parser.add_argument("--peg-depth", type=float, help="Connector peg depth in mm")
    parser.add_argument("--connector-count", type=int, help="Number of connectors per split")
    parser.add_argument("--cut-samples", type=int, help="Samples along axis for cut selection")
    parser.add_argument("--max-depth", type=int, help="Max recursive split depth")
    parser.add_argument("--max-pieces", type=int, help="Max pieces (0 = unlimited)")
    parser.add_argument("--orient-step", type=int, help="Orientation step degrees (e.g., 90)")
    parser.add_argument("--no-repair", action="store_true", help="Skip mesh repair")
    parser.add_argument("--no-orient", action="store_true", help="Skip orientation optimization")
    parser.add_argument("--no-connectors", action="store_true", help="Skip connector generation")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    return parser


def load_default_config(path_override: Optional[Path]) -> AppConfig:
    if path_override:
        return load_config(path_override)
    default_path = Path("config.yaml")
    if default_path.exists():
        return load_config(default_path)
    return AppConfig()


def apply_overrides(config: AppConfig, args: argparse.Namespace) -> AppConfig:
    build = config.build_volume
    if args.build:
        build = BuildVolume(x_mm=args.build[0], y_mm=args.build[1], z_mm=args.build[2])

    connectors = config.connectors
    if args.connector:
        connectors = replace(connectors, style=args.connector)
    if args.tolerance is not None:
        connectors = replace(connectors, tolerance_mm=args.tolerance)
    if args.peg_radius is not None:
        connectors = replace(connectors, peg_radius_mm=args.peg_radius)
    if args.peg_depth is not None:
        connectors = replace(connectors, peg_depth_mm=args.peg_depth)
    if args.connector_count is not None:
        connectors = replace(connectors, count=args.connector_count)

    split = config.split
    if args.cut_samples is not None:
        split = replace(split, cut_samples=args.cut_samples)
    if args.max_depth is not None:
        split = replace(split, max_depth=args.max_depth)
    if args.max_pieces is not None:
        split = replace(split, max_pieces=args.max_pieces)
    if args.orient_step is not None:
        split = replace(split, orient_step_deg=args.orient_step)

    output = config.output
    if args.format:
        output = OutputConfig(format=args.format, engine=output.engine)
    if args.engine:
        output = OutputConfig(format=output.format, engine=args.engine)

    return AppConfig(build_volume=build, connectors=connectors, split=split, output=output)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    config = load_default_config(args.config)
    config = apply_overrides(config, args)

    output_path = args.output
    if output_path is None:
        output_dir = Path("output")
        output_path = output_dir / f"{args.input.stem}.{config.output.format}"

    result = run_pipeline(
        args.input,
        output_path,
        config,
        repair=not args.no_repair,
        orient=not args.no_orient,
        add_connectors=not args.no_connectors and config.connectors.style != "none",
    )

    logging.info(
        "Input: %s verts, %s faces, watertight=%s",
        result.input_stats.vertices,
        result.input_stats.faces,
        result.input_stats.watertight,
    )
    logging.info(
        "Repaired: %s verts, %s faces, watertight=%s",
        result.repaired_stats.vertices,
        result.repaired_stats.faces,
        result.repaired_stats.watertight,
    )
    logging.info("Pieces: %s", result.piece_count)
    for path in result.output_paths:
        logging.info("Wrote: %s", path)


if __name__ == "__main__":
    main()

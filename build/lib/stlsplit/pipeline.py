from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import trimesh

from .analysis import MeshStats, analyze_mesh
from .config import AppConfig
from .export import export_meshes
from .mesh_ops import load_mesh, optimize_orientation, repair_mesh
from .splitter import split_to_fit


@dataclass(frozen=True)
class PipelineResult:
    input_stats: MeshStats
    repaired_stats: MeshStats
    piece_count: int
    output_paths: List[Path]


def run_pipeline(
    input_path: Path,
    output_path: Path,
    config: AppConfig,
    repair: bool = True,
    orient: bool = True,
    add_connectors: bool = True,
) -> PipelineResult:
    mesh = load_mesh(input_path)
    input_stats = analyze_mesh(mesh)

    if repair:
        mesh = repair_mesh(mesh)
    repaired_stats = analyze_mesh(mesh)

    if orient:
        mesh, _ = optimize_orientation(
            mesh, config.build_volume, config.split.orient_step_deg
        )

    pieces = split_to_fit(
        mesh,
        config.build_volume,
        config.split,
        config.connectors,
        config.output.engine,
        add_connectors,
    )

    output_paths = export_meshes(pieces, output_path, config.output.format)

    return PipelineResult(
        input_stats=input_stats,
        repaired_stats=repaired_stats,
        piece_count=len(pieces),
        output_paths=output_paths,
    )

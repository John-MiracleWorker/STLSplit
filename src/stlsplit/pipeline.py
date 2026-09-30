from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import trimesh

from .analysis import MeshStats, analyze_mesh
from .config import AppConfig
from .export import export_meshes
from .mesh_ops import decimate_mesh, load_mesh, optimize_orientation, repair_mesh
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
    repair_mode: str = "light",
    orient: bool = True,
    add_connectors: bool = True,
) -> PipelineResult:
    mesh = load_mesh(input_path)
    
    # Decimate first to save memory
    mesh = decimate_mesh(mesh, config.split.max_vertices)
    
    input_stats = analyze_mesh(mesh)

    if repair_mode != "none":
        mesh = repair_mesh(mesh, mode=repair_mode)
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
        repair_mode=repair_mode,
    )

    # Smart Layout (God-Tier Feature)
    from .layout import arrange_on_plates
    if not pieces:
        # Fallback if no split happened (e.g. fits volume)
        # But even if it fits, we might want to rotate it flat?
        # For now, let's just rotate the single piece if needed.
        pieces = [mesh]

    pieces = arrange_on_plates(pieces, config.build_volume)

    _validate_pieces(pieces, config.build_volume)

    output_files = export_meshes(pieces, output_path, config.output.format)

    return PipelineResult(
        input_stats=input_stats,
        repaired_stats=repaired_stats,
        piece_count=len(pieces),
        output_paths=output_files,
    )


def _validate_pieces(pieces: List, build) -> None:
    """
    Fail loudly on pieces that are unprintable as produced, instead of
    silently writing a 3MF the slicer will reject or that won't fit:
    - not watertight (slicer rejects or prints garbage)
    - footprint exceeds the build volume in the orientation layout placed
      them (layout puts each piece flat, min corner at the origin)
    Exits code 2 on any failure.
    """
    import sys
    build_box = np.array([build.x_mm, build.y_mm, build.z_mm])
    ok = True
    for i, geom in enumerate(pieces):
        if not isinstance(geom, trimesh.Trimesh) or len(geom.faces) == 0:
            continue
        if not geom.is_watertight:
            print(f"❌ piece {i}: NOT watertight — will not slice reliably")
            ok = False
        e = geom.bounding_box.extents
        if not np.all(e <= build_box + 1e-6):
            print(
                f"❌ piece {i}: {np.round(e, 1)} exceeds build volume "
                f"{np.round(build_box, 0)}"
            )
            ok = False
    if ok:
        print(f"✅ validation passed: {len(pieces)} pieces watertight and within build volume")
    else:
        print("Aborting export: fix the pieces above (or change build volume / orientation) before printing.")
        sys.exit(2)

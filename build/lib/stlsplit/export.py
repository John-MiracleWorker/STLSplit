from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

import trimesh


def export_meshes(meshes: Iterable[trimesh.Trimesh], output_path: Path, fmt: str) -> List[Path]:
    fmt = fmt.lower()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if fmt == "3mf":
        scene = trimesh.Scene()
        for idx, mesh in enumerate(meshes):
            scene.add_geometry(mesh, node_name=f"part_{idx:03d}")
        scene.export(file_obj=str(output_path), file_type="3mf")
        return [output_path]

    if fmt == "stl":
        if output_path.suffix:
            out_dir = output_path.parent / output_path.stem
        else:
            out_dir = output_path
        out_dir.mkdir(parents=True, exist_ok=True)
        outputs: List[Path] = []
        for idx, mesh in enumerate(meshes):
            part_path = out_dir / f"part_{idx:03d}.stl"
            mesh.export(part_path)
            outputs.append(part_path)
        return outputs

    raise ValueError(f"Unsupported format: {fmt}")

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import trimesh

from .analysis import face_roughness, plane_band_face_stats
from .config import BuildVolume, ConnectorConfig, SplitConfig
from .connectors import apply_connectors
from .mesh_ops import Plane, fits_build_volume, split_mesh_by_plane


@dataclass(frozen=True)
class SplitDecision:
    plane: Plane
    score: float


def choose_cut_plane(
    mesh: trimesh.Trimesh, build: BuildVolume, config: SplitConfig
) -> Optional[SplitDecision]:
    """
    Intelligently choose a cut plane by sampling positions along the axes
    that exceed the build volume and scoring them based on curvature and density.
    """
    bounds_min, bounds_max = mesh.bounds
    extents = bounds_max - bounds_min
    build_dims = np.array([build.x_mm, build.y_mm, build.z_mm])

    # Find axes that are too large
    oversized_axes = np.where(extents > build_dims)[0]
    if len(oversized_axes) == 0:
        return None

    roughness = face_roughness(mesh)
    best: Optional[SplitDecision] = None
    margin = float(config.cut_margin_mm)

    for axis in oversized_axes:
        normal = np.zeros(3)
        normal[axis] = 1.0

        low = bounds_min[axis] + margin
        high = bounds_max[axis] - margin

        if low < high:
            positions = np.linspace(low, high, max(config.cut_samples, 2))
        else:
            # Fallback to middle if margin is too large or bounds are too close
            positions = np.array([bounds_min[axis] + extents[axis] * 0.5])

        for pos in positions:
            origin = (bounds_min + bounds_max) * 0.5
            origin[axis] = pos

            stats = plane_band_face_stats(
                mesh, roughness, origin, normal, config.curvature_band_mm
            )

            # Supportability: Cuts that create large flat areas are good bases.
            # stats['area'] should be added to plane_band_face_stats or calculated here.
            # For now let's use a simpler heuristic or update analysis.py
            area = stats.get("area", 0.0)
            
            # Score: we want LOW roughness, LOW density, and LARGE area
            score = (
                config.roughness_weight * stats["roughness"]
                + config.density_weight * stats["density"]
                - (config.support_weight * (area / (extents.max()**2 + 1e-6)))
            )

            decision = SplitDecision(plane=Plane(origin=origin, normal=normal), score=float(score))
            if best is None or decision.score < best.score:
                best = decision

    return best


def split_to_fit(
    mesh: trimesh.Trimesh,
    build: BuildVolume,
    split_cfg: SplitConfig,
    connector_cfg: ConnectorConfig,
    engine: str,
    add_connectors: bool,
) -> List[trimesh.Trimesh]:
    pieces: List[trimesh.Trimesh] = []
    queue: List[Tuple[trimesh.Trimesh, int]] = [(mesh, 0)]

    while queue:
        current, depth = queue.pop()
        if fits_build_volume(current, build) or depth >= split_cfg.max_depth:
            pieces.append(current)
            continue
        if split_cfg.max_pieces and (len(pieces) + len(queue) + 1) >= split_cfg.max_pieces:
            pieces.append(current)
            continue

        decision = choose_cut_plane(current, build, split_cfg)
        if not decision:
            pieces.append(current)
            continue

        pos_mesh, neg_mesh = split_mesh_by_plane(
            current, decision.plane.origin, decision.plane.normal
        )
        if pos_mesh is None or neg_mesh is None:
            pieces.append(current)
            continue
        if len(pos_mesh.faces) == 0 or len(neg_mesh.faces) == 0:
            pieces.append(current)
            continue

        if add_connectors:
            try:
                # Create a unique label for this connection
                label_str = f"P{depth}_{len(pieces)}"
                pos_mesh, neg_mesh = apply_connectors(
                    pos_mesh,
                    neg_mesh,
                    decision.plane.origin,
                    decision.plane.normal,
                    connector_cfg,
                    engine,
                    label=label_str,
                )
            except Exception as e:
                # Fallback to no connectors if boolean fails
                print(f"Warning: Connector failed ({e}), skipping connectors for this split.")

        queue.append((pos_mesh, depth + 1))
        queue.append((neg_mesh, depth + 1))

    return pieces

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import trimesh


@dataclass(frozen=True)
class MeshStats:
    vertices: int
    faces: int
    watertight: bool
    volume_mm3: float
    bounds_min: np.ndarray
    bounds_max: np.ndarray

    @property
    def extents_mm(self) -> np.ndarray:
        return self.bounds_max - self.bounds_min


def analyze_mesh(mesh: trimesh.Trimesh) -> MeshStats:
    bounds_min, bounds_max = mesh.bounds
    volume = float(mesh.volume) if mesh.is_watertight else 0.0
    return MeshStats(
        vertices=int(len(mesh.vertices)),
        faces=int(len(mesh.faces)),
        watertight=bool(mesh.is_watertight),
        volume_mm3=volume,
        bounds_min=bounds_min,
        bounds_max=bounds_max,
    )


def face_roughness(mesh: trimesh.Trimesh) -> np.ndarray:
    if len(mesh.faces) == 0:
        return np.array([])
    adjacency = mesh.face_adjacency
    angles = mesh.face_adjacency_angles
    roughness = np.zeros(len(mesh.faces), dtype=float)
    counts = np.zeros(len(mesh.faces), dtype=float)
    for (f1, f2), angle in zip(adjacency, angles):
        roughness[f1] += angle
        roughness[f2] += angle
        counts[f1] += 1.0
        counts[f2] += 1.0
    roughness = np.divide(roughness, counts, out=np.zeros_like(roughness), where=counts > 0)
    return roughness


def face_gaussian_curvature(mesh: trimesh.Trimesh, radius_mm: float) -> np.ndarray:
    if len(mesh.faces) == 0 or len(mesh.vertices) == 0:
        return np.array([])
    radius = max(float(radius_mm), 1e-6)

    curvature_fn = getattr(trimesh.curvature, "smoothing_gaussian", None)
    vertex_curvature = None
    if curvature_fn is not None:
        try:
            vertex_curvature = curvature_fn(mesh, mesh.vertices, radius)
        except Exception:
            try:
                vertex_curvature = curvature_fn(mesh, radius)
            except Exception:
                vertex_curvature = None

    if vertex_curvature is None:
        try:
            vertex_curvature = trimesh.curvature.discrete_gaussian_curvature_measure(
                mesh, mesh.vertices, radius
            )
        except Exception:
            return np.zeros(len(mesh.faces), dtype=float)

    if len(vertex_curvature) != len(mesh.vertices):
        return np.zeros(len(mesh.faces), dtype=float)

    face_curvature = np.abs(vertex_curvature[mesh.faces]).mean(axis=1)
    face_curvature = np.nan_to_num(face_curvature, nan=0.0, posinf=0.0, neginf=0.0)

    scale = np.percentile(face_curvature, 90) if len(face_curvature) else 0.0
    if scale > 0:
        face_curvature = face_curvature / scale

    return face_curvature


def plane_band_face_stats(
    mesh: trimesh.Trimesh,
    roughness: np.ndarray,
    plane_origin: np.ndarray,
    plane_normal: np.ndarray,
    band_mm: float,
    curvature: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    centers = mesh.triangles_center
    distances = np.dot(centers - plane_origin, plane_normal)
    mask = np.abs(distances) <= band_mm
    
    # Calculate cross-sectional area for supportability scoring
    try:
        section = mesh.section(plane_origin=plane_origin, plane_normal=plane_normal)
        area = float(section.area) if section else 0.0
    except Exception:
        area = 0.0

    if not np.any(mask):
        return {"roughness": 1.0, "density": 1.0, "area": area, "curvature": 1.0}
    
    band_roughness = float(np.median(roughness[mask])) if len(roughness) else 1.0
    density = float(np.count_nonzero(mask) / max(len(mesh.faces), 1))
    if curvature is not None and len(curvature) == len(mesh.faces):
        band_curvature = float(np.median(curvature[mask]))
    else:
        band_curvature = 1.0
    return {
        "roughness": band_roughness,
        "density": density,
        "area": area,
        "curvature": band_curvature,
    }

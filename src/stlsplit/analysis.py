from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

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


def plane_band_face_stats(
    mesh: trimesh.Trimesh,
    roughness: np.ndarray,
    plane_origin: np.ndarray,
    plane_normal: np.ndarray,
    band_mm: float,
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
        return {"roughness": 1.0, "density": 1.0, "area": area}
    
    band_roughness = float(np.median(roughness[mask])) if len(roughness) else 1.0
    density = float(np.count_nonzero(mask) / max(len(mesh.faces), 1))
    return {"roughness": band_roughness, "density": density, "area": area}

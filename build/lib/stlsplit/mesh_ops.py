from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import trimesh

from .config import BuildVolume


@dataclass(frozen=True)
class Plane:
    origin: np.ndarray
    normal: np.ndarray


def load_mesh(path: Path) -> trimesh.Trimesh:
    mesh = trimesh.load(path, force="mesh", process=False)
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError(f"Unsupported mesh at {path}")
    return mesh


def repair_mesh(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """
    Attempts to make a mesh watertight and manifold.
    """
    fixed = mesh.copy()
    
    # trimesh repair functions
    try:
        trimesh.repair.fix_inversion(fixed)
        trimesh.repair.fix_winding(fixed)
        trimesh.repair.fix_normals(fixed)
        trimesh.repair.fill_holes(fixed)
    except Exception:
        pass
        
    # Process removes NaNs, merges vertices, and if validate=True:
    # removes degenerate/duplicate faces and ensures consistent winding.
    fixed.process(validate=True)
    return fixed


def fits_build_volume(mesh: trimesh.Trimesh, build: BuildVolume) -> bool:
    """ Checks if mesh fits anywhere in the build volume. """
    extents = mesh.extents
    # Build volume is X, Y, Z. We can rotate the object to fit.
    # So we check if the sorted extents are <= sorted build dimensions.
    return np.all(np.sort(extents) <= np.sort([build.x_mm, build.y_mm, build.z_mm]))


def iter_rotations(step_deg: int) -> Iterable[np.ndarray]:
    """ Generates 4x4 rotation matrices for sampling. """
    step = max(int(step_deg), 1)
    # Optimization: 90 degree increments are often enough for rectangular volumes
    angles = list(range(0, 360, step))
    for rx in angles:
        for ry in angles:
            for rz in angles:
                yield trimesh.transformations.euler_matrix(
                    np.radians(rx), np.radians(ry), np.radians(rz), "sxyz"
                )


def best_fit_transform(mesh: trimesh.Trimesh, build: BuildVolume, step_deg: int) -> np.ndarray:
    """
    Finds the orientation that best fits the build volume.
    Score is based on how well the extents fit within the volume.
    """
    best_score = -np.inf
    best_transform = np.eye(4)
    target = np.sort([build.x_mm, build.y_mm, build.z_mm])
    
    for transform in iter_rotations(step_deg):
        rotated_extents = mesh.copy()
        rotated_extents.apply_transform(transform)
        extents = np.sort(rotated_extents.extents)
        
        # Ratio of build volume to mesh extents (higher is better)
        ratios = target / np.maximum(extents, 1e-6)
        score = float(np.min(ratios))
        
        if score > best_score:
            best_score = score
            best_transform = transform
            if score >= 1.0: # Fits perfectly
                break
                
    return best_transform


def optimize_orientation(
    mesh: trimesh.Trimesh, build: BuildVolume, step_deg: int
) -> Tuple[trimesh.Trimesh, np.ndarray]:
    transform = best_fit_transform(mesh, build, step_deg)
    oriented = mesh.copy()
    oriented.apply_transform(transform)
    return oriented, transform


def split_mesh_by_plane(
    mesh: trimesh.Trimesh, plane_origin: np.ndarray, plane_normal: np.ndarray
) -> Tuple[trimesh.Trimesh, trimesh.Trimesh]:
    positive = trimesh.intersections.slice_mesh_plane(
        mesh, plane_normal=plane_normal, plane_origin=plane_origin, cap=True
    )
    negative = trimesh.intersections.slice_mesh_plane(
        mesh, plane_normal=-plane_normal, plane_origin=plane_origin, cap=True
    )
    return positive, negative


def boolean_op(
    meshes: Iterable[trimesh.Trimesh], op: str, engine: str
) -> trimesh.Trimesh:
    if op == "union":
        return trimesh.boolean.union(list(meshes), engine=engine)
    if op == "difference":
        meshes = list(meshes)
        if len(meshes) != 2:
            raise ValueError("difference expects exactly two meshes")
        return trimesh.boolean.difference(meshes, engine=engine)
    if op == "intersection":
        return trimesh.boolean.intersection(list(meshes), engine=engine)
    raise ValueError(f"Unknown boolean op: {op}")


def plane_basis(normal: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = normal / np.linalg.norm(normal)
    ref = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(ref, n)) > 0.9:
        ref = np.array([0.0, 1.0, 0.0])
    u = np.cross(n, ref)
    u /= np.linalg.norm(u)
    v = np.cross(n, u)
    return n, u, v


def rotation_from_z(normal: np.ndarray) -> np.ndarray:
    n = normal / np.linalg.norm(normal)
    z = np.array([0.0, 0.0, 1.0])
    if np.allclose(n, z):
        return np.eye(4)
    if np.allclose(n, -z):
        return trimesh.transformations.rotation_matrix(np.pi, [1.0, 0.0, 0.0])
    v = np.cross(z, n)
    s = np.linalg.norm(v)
    c = float(np.dot(z, n))
    vx = np.array(
        [
            [0.0, -v[2], v[1]],
            [v[2], 0.0, -v[0]],
            [-v[1], v[0], 0.0],
        ]
    )
    r = np.eye(3) + vx + (vx @ vx) * ((1.0 - c) / (s * s))
    transform = np.eye(4)
    transform[:3, :3] = r
    return transform


def embossed_text_mesh(text: str, height: float, font: Optional[str] = None) -> trimesh.Trimesh:
    """
    Creates a 3D mesh of the given text.
    """
    try:
        t_mesh = trimesh.creation.text_mesh(text, font=font)
        # Scale to desired height (Y is usually up in text mesh generation)
        h = t_mesh.extents[1] 
        scale = height / h if h > 0 else 1.0
        t_mesh.apply_scale(scale)
        # Deep repair because text meshes can be messy
        t_mesh.process(validate=True)
        return t_mesh
    except Exception as e:
        # Fallback for headless environments or missing fonts
        return trimesh.creation.box(extents=[height, height, 2.0])

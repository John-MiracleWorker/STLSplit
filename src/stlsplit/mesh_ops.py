from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple

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


def repair_mesh(mesh: trimesh.Trimesh, mode: str = "light") -> trimesh.Trimesh:
    """
    Attempts to make a mesh watertight and manifold.
    Uses pymeshfix for aggressive repair if available.
    """
    fixed = mesh.copy()
    if mode == "none":
        return fixed

    if mode == "aggressive":
        # Try pymeshfix first (most aggressive repair)
        try:
            import pymeshfix
            import pyvista as pv

            # Convert trimesh to pyvista
            faces_with_counts = np.hstack(
                [np.full((len(fixed.faces), 1), 3, dtype=np.int64), fixed.faces]
            )
            pv_mesh = pv.PolyData(fixed.vertices, faces_with_counts)

            # Run pymeshfix
            meshfix = pymeshfix.MeshFix(pv_mesh)
            meshfix.repair(verbose=False)
            repaired_pv = meshfix.mesh

            # Convert back to trimesh
            if repaired_pv is not None and repaired_pv.n_cells > 0:
                # Extract faces from pyvista format
                faces_flat = repaired_pv.faces
                # Reshape: pyvista stores [n_verts, v0, v1, v2, n_verts, v0, v1, v2, ...]
                faces = faces_flat.reshape(-1, 4)[:, 1:4]
                fixed = trimesh.Trimesh(vertices=repaired_pv.points, faces=faces)
                fixed.process(validate=True)
                if fixed.is_volume:
                    return fixed
        except Exception:
            pass  # Fall back to standard repair

        # Try fill holes if still non-volume (aggressive, but cheaper than hull)
        if not fixed.is_volume:
            try:
                fixed = mesh.copy()
                fixed.fill_holes()
                fixed.process(validate=True)
                if fixed.is_volume:
                    return fixed
            except Exception:
                pass

    # Standard trimesh repair (lightweight)
    fixed = mesh.copy()
    try:
        fixed.remove_infinite_values()
        fixed.remove_unreferenced_vertices()
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
    mesh: trimesh.Trimesh,
    plane_origin: np.ndarray,
    plane_normal: np.ndarray,
    repair_mode: str = "light",
) -> Tuple[trimesh.Trimesh, trimesh.Trimesh]:
    tri_engine = pick_triangulation_engine()
    positive = trimesh.intersections.slice_mesh_plane(
        mesh,
        plane_normal=plane_normal,
        plane_origin=plane_origin,
        cap=True,
        engine=tri_engine,
    )
    negative = trimesh.intersections.slice_mesh_plane(
        mesh,
        plane_normal=-plane_normal,
        plane_origin=plane_origin,
        cap=True,
        engine=tri_engine,
    )
    
    # Use aggressive repair to ensure watertight volumes
    if positive is not None:
        positive = repair_mesh(positive, mode=repair_mode)
    if negative is not None:
        negative = repair_mesh(negative, mode=repair_mode)
        
    return positive, negative


def pick_triangulation_engine(
    preferred: Optional[Sequence[str]] = None,
) -> Optional[str]:
    preferred = preferred or ("earcut", "manifold", "triangle")
    engines = dict(getattr(trimesh.creation, "_engines", []))
    for name in preferred:
        if engines.get(name):
            return name
    return None


def boolean_op(
    meshes: Iterable[trimesh.Trimesh], op: str, engine: str
) -> trimesh.Trimesh:
    engines_available = getattr(trimesh.boolean, "engines_available", set())
    engines_to_try = []
    if engine in engines_available:
        engines_to_try.append(engine)
    if engine not in engines_available and engine is not None:
        engines_to_try.append(None)
    for candidate in engines_available:
        if candidate not in engines_to_try:
            engines_to_try.append(candidate)

    if op == "union":
        last_error = None
        for candidate in engines_to_try:
            try:
                return trimesh.boolean.union(list(meshes), engine=candidate)
            except Exception as e:
                last_error = e
        raise last_error or ValueError("No boolean engine available for union")
    if op == "difference":
        meshes = list(meshes)
        if len(meshes) != 2:
            raise ValueError("difference expects exactly two meshes")
        last_error = None
        for candidate in engines_to_try:
            try:
                return trimesh.boolean.difference(meshes, engine=candidate)
            except Exception as e:
                last_error = e
        raise last_error or ValueError("No boolean engine available for difference")
    if op == "intersection":
        last_error = None
        for candidate in engines_to_try:
            try:
                return trimesh.boolean.intersection(list(meshes), engine=candidate)
            except Exception as e:
                last_error = e
        raise last_error or ValueError("No boolean engine available for intersection")
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
        if not t_mesh.is_volume:
            try:
                t_mesh = t_mesh.convex_hull
            except Exception:
                t_mesh = None
        if t_mesh is None or not t_mesh.is_volume:
            return trimesh.creation.box(extents=[height, height, 2.0])
        return t_mesh
    except Exception:
        # Fallback for headless environments or missing fonts
        return trimesh.creation.box(extents=[height, height, 2.0])

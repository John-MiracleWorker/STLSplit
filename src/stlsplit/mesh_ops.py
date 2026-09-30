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



def decimate_mesh(mesh: trimesh.Trimesh, max_vertices: int) -> trimesh.Trimesh:
    """
    Simplifies the mesh if it exceeds the max_vertices count.
    Uses quadratic decimation.
    """
    if len(mesh.vertices) <= max_vertices:
        return mesh

    print(f"📉 Decimating mesh from {len(mesh.vertices)} to {max_vertices} vertices...")
    target_faces = int(max_vertices * 2)  # Approx relationship
    try:
        simplified = mesh.simplify_quadric_decimation(target_faces)
        print(f"📉 Decimated to {len(simplified.vertices)} vertices.")
        return simplified
    except Exception as e:
        print(f"⚠️ Decimation failed: {e}")
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
    """
    Check if mesh fits the build volume in its print orientation.

    Parts are printed flat: layout.orient_to_bed puts the SMALLEST extent
    vertical (bed axis), so the footprint is the two larger extents. The
    footprint may be rotated 90° on the bed.

    NOTE: the old implementation compared sorted extents, which treated the
    build box as if the part could be rotated into ANY orientation (e.g. a
    181x246x60 part "fit" because 246 < 250 when stood on end). But the
    layout always prints flat, so such parts ended up 246mm wide on a 220mm
    plate. This version checks the orientation the part is actually printed
    in, so "fits" always means "printable as arranged".
    """
    extents = mesh.extents
    order = np.argsort(extents)
    height = extents[order[0]]
    f1 = extents[order[1]]
    f2 = extents[order[2]]
    if height > build.z_mm:
        return False
    tol = 1e-6
    return (f1 <= build.x_mm + tol and f2 <= build.y_mm + tol) or (
        f2 <= build.x_mm + tol and f1 <= build.y_mm + tol
    )


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


def _manifold_to_trimesh(m) -> Optional[trimesh.Trimesh]:
    """Convert a manifold3d.Manifold to a trimesh.Trimesh (None if empty)."""
    if m is None or m.is_empty():
        return None
    raw = m.to_mesh()
    faces = np.asarray(raw.tri_verts, dtype=np.int64)
    verts = np.asarray(raw.vert_properties, dtype=np.float64)[:, :3]
    if len(faces) == 0 or len(verts) == 0:
        return None
    return trimesh.Trimesh(vertices=verts, faces=faces, process=True)


def _split_mesh_manifold(
    mesh: trimesh.Trimesh,
    plane_origin: np.ndarray,
    plane_normal: np.ndarray,
) -> Optional[Tuple[trimesh.Trimesh, trimesh.Trimesh]]:
    """
    Split via manifold3d CSG (split_by_plane). Constructive solid geometry
    guarantees watertight children, unlike slice_mesh_plane whose capped
    slices sometimes leave holes fill_holes cannot close (observed on
    2nd-generation cuts of complex parts -> 'Not all meshes are volumes!'
    in connector booleans).
    Returns (positive_side, negative_side) or None if manifold is unusable.
    The positive side is the one in the direction of plane_normal.
    """
    try:
        import manifold3d as m3d
        normal = np.asarray(plane_normal, dtype=np.float64)
        origin = np.asarray(plane_origin, dtype=np.float64)
        mm = m3d.Manifold(
            m3d.Mesh(
                np.ascontiguousarray(np.asarray(mesh.vertices, dtype=np.float32)),
                np.ascontiguousarray(np.asarray(mesh.faces, dtype=np.uint32)),
            )
        )
        first, second = mm.split_by_plane(normal, float(normal @ origin))
        positive = _manifold_to_trimesh(first)
        negative = _manifold_to_trimesh(second)
        if positive is not None and negative is not None:
            return positive, negative
        return None
    except Exception as e:
        import traceback
        print(f"DEBUG: manifold split failed: {e!r}")
        traceback.print_exc(limit=3)
        return None


def split_mesh_by_plane(
    mesh: trimesh.Trimesh,
    plane_origin: np.ndarray,
    plane_normal: np.ndarray,
    repair_mode: str = "light",
) -> Tuple[trimesh.Trimesh, trimesh.Trimesh]:
    # Preferred path: manifold3d CSG split -> watertight children guaranteed.
    manifold_result = _split_mesh_manifold(mesh, plane_origin, plane_normal)
    if manifold_result is not None:
        positive, negative = manifold_result
        if positive.is_watertight and negative.is_watertight:
            return positive, negative
        # manifold produced non-watertight (shouldn't happen); fall through
        # to the slice path which may repair differently.

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
    meshes: Iterable[trimesh.Trimesh],
    op: str,
    engine: str,
    check_volume: bool = True,
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
                return trimesh.boolean.union(
                    list(meshes), engine=candidate, check_volume=check_volume
                )
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
                return trimesh.boolean.difference(
                    meshes, engine=candidate, check_volume=check_volume
                )
            except Exception as e:
                last_error = e
        raise last_error or ValueError("No boolean engine available for difference")
    if op == "intersection":
        last_error = None
        for candidate in engines_to_try:
            try:
                return trimesh.boolean.intersection(
                    list(meshes), engine=candidate, check_volume=check_volume
                )
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

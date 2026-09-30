from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import trimesh

from .config import ConnectorConfig
from .mesh_ops import (
    boolean_op,
    embossed_text_mesh,
    pick_triangulation_engine,
    plane_basis,
    repair_mesh,
    rotation_from_z,
)


def connector_positions(
    section_vertices: np.ndarray,
    plane_origin: np.ndarray,
    plane_normal: np.ndarray,
    connector_count: int,
    margin_mm: float,
) -> List[np.ndarray]:
    """
    Calculate peg positions based on the 2D cross-section of the cut.
    Returns positions ON the cut plane.
    """
    if section_vertices is None or len(section_vertices) == 0:
        print(f"DEBUG: No section vertices, using plane_origin: {plane_origin}")
        return [plane_origin.copy()]

    _, u, v = plane_basis(plane_normal)
    
    # Project vertices onto the plane's 2D coordinate system
    coords = np.vstack(
        (
            np.dot(section_vertices - plane_origin, u),
            np.dot(section_vertices - plane_origin, v),
        )
    ).T
    
    if not np.isfinite(coords).all():
        print(f"DEBUG: Non-finite coords, using plane_origin")
        return [plane_origin.copy()]

    min_uv = coords.min(axis=0)
    max_uv = coords.max(axis=0)
    extents_uv = max_uv - min_uv
    center_uv = (min_uv + max_uv) * 0.5
    
    print(f"DEBUG: Section extents UV: {extents_uv}, center UV: {center_uv}")

    # Choose the longer axis for spreading multiple connectors
    if extents_uv[0] >= extents_uv[1]:
        axis_idx = 0
    else:
        axis_idx = 1

    max_offset = (extents_uv[axis_idx] * 0.5) - margin_mm
    
    if connector_count <= 1 or max_offset <= 0:
        # Just one in the center
        pos = plane_origin + u * center_uv[0] + v * center_uv[1]
        print(f"DEBUG: Single connector position: {pos}")
        return [pos]

    # Spread along the chosen axis
    offsets = np.linspace(-max_offset, max_offset, connector_count)
    
    results = []
    for off in offsets:
        if axis_idx == 0:
            p = plane_origin + u * (center_uv[0] + off) + v * center_uv[1]
        else:
            p = plane_origin + u * center_uv[0] + v * (center_uv[1] + off)
        results.append(p)
    
    print(f"DEBUG: Generated {len(results)} connector positions")
    return results


def make_hex_prism(radius: float, height: float) -> trimesh.Trimesh:
    """Creates a hex prism (6-sided cylinder) aligned to Z axis."""
    return trimesh.creation.cylinder(radius=radius, height=height, sections=6)


def make_dovetail(width: float, height: float, depth: float, angle_deg: float) -> trimesh.Trimesh:
    """
    Creates a dovetail wedge.
    width: bottom (widest) width
    height: total thickness of the wedge
    depth: how far it extends into the part
    angle_deg: angle of the sides
    """
    angle = np.radians(angle_deg)
    top_width = width - 2.0 * depth * np.tan(angle)
    if top_width <= 0:
        top_width = width * 0.5

    # Re-oriented to point along Z+ (Depth is Z)
    # This ensures it aligns with the cut normal correctly
    # x - width, y - thickness (height arg), z - depth
    
    # Vertices
    # Base at Z=0
    v_base = [
        [-width/2, -height/2, 0], 
        [width/2, -height/2, 0], 
        [width/2, height/2, 0], 
        [-width/2, height/2, 0]
    ]
    # Tip at Z=depth
    v_tip = [
        [-top_width/2, -height/2, depth], 
        [top_width/2, -height/2, depth], 
        [top_width/2, height/2, depth], 
        [-top_width/2, height/2, depth]
    ]
    v = np.array(v_base + v_tip)

    f = np.array([
        [0, 2, 1], [0, 3, 2], # Bottom (Base)
        [4, 5, 6], [4, 6, 7], # Top (Tip)
        [0, 1, 5], [0, 5, 4], # Side 1 (Front?)
        [1, 2, 6], [1, 6, 5], # Side 2 (Right)
        [2, 3, 7], [2, 7, 6], # Side 3 (Back?)
        [3, 0, 4], [3, 4, 7]  # Side 4 (Left)
    ])
    return trimesh.Trimesh(vertices=v, faces=f)


def is_thin_walled_section(
    section_vertices: np.ndarray,
    section_path: Optional[trimesh.path.Path3D],
    section_area: float,
    threshold: float = 0.15,
) -> bool:
    """
    Detects thin-walled/shell sections using compactness ratio (area / perimeter^2).
    
    Thin shells (like helmet cross-sections) have high perimeter relative to area.
    This is the "isoperimetric quotient" - circles are most compact (~0.08).
    
    Args:
        section_vertices: 3D vertices of the cross-section
        section_path: Optional path object with perimeter info
        section_area: Area of the cross-section
        threshold: Compactness below this = thin-walled (default 0.15)
    
    Returns:
        True if the section appears to be thin-walled/shell geometry
    
    Reference values:
        - Circle: ~0.08 (most compact)
        - Square: ~0.0625
        - Thin ring/shell: ~0.01-0.03
        - Very thin shell: < 0.01
    """
    if section_area <= 0:
        return False
    
    perimeter = 0.0
    
    # Try to get perimeter from path first (most accurate)
    if section_path is not None:
        try:
            perimeter = section_path.length
        except Exception:
            pass
    
    # Fallback: estimate perimeter from vertices
    if perimeter <= 0 and section_vertices is not None and len(section_vertices) >= 3:
        try:
            # Calculate perimeter as sum of edge lengths
            verts = _finite_vertices(section_vertices)
            if len(verts) >= 3:
                # Close the loop
                edges = np.diff(np.vstack([verts, verts[0:1]]), axis=0)
                perimeter = float(np.sum(np.linalg.norm(edges, axis=1)))
        except Exception:
            pass
    
    if perimeter <= 0:
        return False
    
    # Compactness = area / perimeter^2
    # Lower values = more elongated/thin shape
    compactness = section_area / (perimeter * perimeter)
    
    print(f"DEBUG: Section compactness={compactness:.4f} (area={section_area:.1f}, perimeter={perimeter:.1f})")
    
    return compactness < threshold


def make_lip_groove(
    section_path: Optional[trimesh.path.Path3D],
    section_vertices: np.ndarray,
    plane_origin: np.ndarray,
    plane_normal: np.ndarray,
    lip_height: float,
    lip_width: float,
    tolerance: float,
    engine: Optional[str] = None,
) -> Tuple[Optional[trimesh.Trimesh], Optional[trimesh.Trimesh]]:
    """
    Creates a lip (protrusion) and groove (recess) that follows the section profile.
    
    For thin-walled models like helmets, this creates an interlocking edge
    that doesn't require boolean operations to punch through walls.
    
    The lip is a thin extrusion along the outer edge of the section.
    The groove is the matching recess with tolerance added.
    
    Args:
        section_path: Path3D of the cross-section (preferred)
        section_vertices: Vertices of the cross-section (fallback)
        plane_origin: Origin point of the cut plane
        plane_normal: Normal vector of the cut plane
        lip_height: How far the lip protrudes from the cut plane
        lip_width: Thickness/width of the lip
        tolerance: Gap between lip and groove for fit
        engine: Triangulation engine to use
    
    Returns:
        (lip_mesh, groove_mesh) or (None, None) on failure
    """
    from shapely.geometry import MultiPoint, LineString
    from shapely.ops import unary_union
    from shapely import buffer
    
    try:
        _, u, v = plane_basis(plane_normal)
        
        # Get 2D representation of section
        poly_2d = None
        
        if section_path is not None:
            try:
                planar, to_3d = section_path.to_2D(normal=plane_normal)
                polygons = list(getattr(planar, "polygons_full", []))
                if polygons:
                    poly_2d = unary_union(polygons)
            except Exception:
                pass
        
        # Fallback: use convex hull of vertices
        if poly_2d is None and section_vertices is not None and len(section_vertices) >= 3:
            verts = _finite_vertices(section_vertices)
            if len(verts) >= 3:
                coords = np.vstack((
                    np.dot(verts - plane_origin, u),
                    np.dot(verts - plane_origin, v),
                )).T
                if np.isfinite(coords).all():
                    poly_2d = MultiPoint(coords).convex_hull
        
        if poly_2d is None or poly_2d.is_empty:
            return None, None
        
        # Create lip as a thin ring around the outer boundary
        # Positive buffer then difference = ring shape
        outer = buffer(poly_2d, lip_width)
        lip_ring = outer.difference(poly_2d)
        
        if lip_ring.is_empty or lip_ring.area <= 0:
            return None, None
        
        # Create groove with tolerance
        outer_groove = buffer(poly_2d, lip_width + tolerance)
        groove_ring = outer_groove.difference(buffer(poly_2d, -tolerance))
        
        if groove_ring.is_empty:
            groove_ring = lip_ring  # Fallback
        
        # Handle MultiPolygon - extrude each polygon separately and combine
        from shapely.geometry import MultiPolygon, Polygon
        
        def extrude_geometry(geom, height, engine):
            """Extrude a Polygon or MultiPolygon, handling both cases."""
            meshes = []
            if isinstance(geom, MultiPolygon):
                for poly in geom.geoms:
                    if not poly.is_empty and poly.area > 0:
                        try:
                            m = trimesh.creation.extrude_polygon(poly, height=height, engine=engine)
                            if m is not None and len(m.vertices) > 0:
                                meshes.append(m)
                        except Exception:
                            pass
            elif isinstance(geom, Polygon):
                if not geom.is_empty and geom.area > 0:
                    try:
                        m = trimesh.creation.extrude_polygon(geom, height=height, engine=engine)
                        if m is not None and len(m.vertices) > 0:
                            meshes.append(m)
                    except Exception:
                        pass
            
            if not meshes:
                return None
            if len(meshes) == 1:
                return meshes[0]
            return trimesh.util.concatenate(meshes)
        
        # Extrude lip and groove
        lip_mesh = extrude_geometry(lip_ring, lip_height, engine)
        groove_mesh = extrude_geometry(groove_ring, lip_height + tolerance * 2, engine)
        
        if lip_mesh is None or groove_mesh is None:
            print("DEBUG: Failed to extrude lip or groove geometry")
            return None, None
        
        # Transform back to 3D space
        # Build transform: 2D coords are in UV space, need to go to world
        if section_path is not None:
            try:
                _, to_3d = section_path.to_2D(normal=plane_normal)
                lip_mesh.apply_transform(to_3d)
                groove_mesh.apply_transform(to_3d)
            except Exception:
                # Manual transform
                transform = np.eye(4)
                transform[:3, 0] = u
                transform[:3, 1] = v
                transform[:3, 2] = plane_normal / np.linalg.norm(plane_normal)
                transform[:3, 3] = plane_origin
                lip_mesh.apply_transform(transform)
                groove_mesh.apply_transform(transform)
        else:
            transform = np.eye(4)
            transform[:3, 0] = u
            transform[:3, 1] = v
            transform[:3, 2] = plane_normal / np.linalg.norm(plane_normal)
            transform[:3, 3] = plane_origin
            lip_mesh.apply_transform(transform)
            groove_mesh.apply_transform(transform)
        
        # Clean up meshes
        lip_mesh = _clean_mesh(lip_mesh)
        groove_mesh = _clean_mesh(groove_mesh)
        
        print(f"DEBUG: Created lip ({lip_mesh.vertices.shape[0]} verts) and groove ({groove_mesh.vertices.shape[0]} verts)")
        
        return lip_mesh, groove_mesh
        
    except Exception as e:
        print(f"DEBUG: make_lip_groove failed: {e}")
        return None, None


def make_seam_plate(
    section_path: Optional[trimesh.path.Path3D],
    section_vertices: np.ndarray,
    plane_origin: np.ndarray,
    plane_normal: np.ndarray,
    plate_thickness: float,
    plate_width: float,
    engine: Optional[str] = None,
) -> Optional[trimesh.Trimesh]:
    """
    Creates a thin internal seam plate that follows the cut profile.
    The plate is a ring inset from the section boundary by plate_width.
    """
    from shapely.geometry import MultiPoint, MultiPolygon, Polygon
    from shapely.ops import unary_union
    from shapely import buffer

    try:
        _, u, v = plane_basis(plane_normal)

        poly_2d = None
        if section_path is not None:
            try:
                planar, _ = section_path.to_2D(normal=plane_normal)
                polygons = list(getattr(planar, "polygons_full", []))
                if polygons:
                    poly_2d = unary_union(polygons)
            except Exception:
                pass

        if poly_2d is None and section_vertices is not None and len(section_vertices) >= 3:
            verts = _finite_vertices(section_vertices)
            if len(verts) >= 3:
                coords = np.vstack(
                    (
                        np.dot(verts - plane_origin, u),
                        np.dot(verts - plane_origin, v),
                    )
                ).T
                if np.isfinite(coords).all():
                    poly_2d = MultiPoint(coords).convex_hull

        if poly_2d is None or poly_2d.is_empty:
            return None

        inner = buffer(poly_2d, -plate_width)
        if inner.is_empty:
            inner = buffer(poly_2d, -plate_width * 0.5)
        if inner.is_empty:
            return None

        ring = poly_2d.difference(inner)
        if ring.is_empty or ring.area <= 0:
            return None

        def extrude_geometry(geom, height, engine):
            meshes = []
            if isinstance(geom, MultiPolygon):
                for poly in geom.geoms:
                    if not poly.is_empty and poly.area > 0:
                        try:
                            m = trimesh.creation.extrude_polygon(
                                poly, height=height, engine=engine
                            )
                            if m is not None and len(m.vertices) > 0:
                                meshes.append(m)
                        except Exception:
                            pass
            elif isinstance(geom, Polygon):
                if not geom.is_empty and geom.area > 0:
                    try:
                        m = trimesh.creation.extrude_polygon(
                            geom, height=height, engine=engine
                        )
                        if m is not None and len(m.vertices) > 0:
                            meshes.append(m)
                    except Exception:
                        pass

            if not meshes:
                return None
            if len(meshes) == 1:
                return meshes[0]
            return trimesh.util.concatenate(meshes)

        plate_mesh = extrude_geometry(ring, plate_thickness, engine)
        if plate_mesh is None:
            return None

        if section_path is not None:
            try:
                _, to_3d = section_path.to_2D(normal=plane_normal)
                plate_mesh.apply_transform(to_3d)
            except Exception:
                transform = np.eye(4)
                transform[:3, 0] = u
                transform[:3, 1] = v
                transform[:3, 2] = plane_normal / np.linalg.norm(plane_normal)
                transform[:3, 3] = plane_origin
                plate_mesh.apply_transform(transform)
        else:
            transform = np.eye(4)
            transform[:3, 0] = u
            transform[:3, 1] = v
            transform[:3, 2] = plane_normal / np.linalg.norm(plane_normal)
            transform[:3, 3] = plane_origin
            plate_mesh.apply_transform(transform)

        plate_mesh.apply_translation(-plane_normal * (plate_thickness * 0.5))
        plate_mesh = _clean_mesh(plate_mesh)
        return plate_mesh

    except Exception as e:
        print(f"DEBUG: make_seam_plate failed: {e}")
        return None

def _section_bbox_area(
    section_vertices: np.ndarray,
    plane_origin: np.ndarray,
    plane_normal: np.ndarray,
) -> float:
    if section_vertices is None or len(section_vertices) < 3:
        return 0.0

    _, u, v = plane_basis(plane_normal)
    coords = np.vstack(
        (
            np.dot(section_vertices - plane_origin, u),
            np.dot(section_vertices - plane_origin, v),
        )
    ).T
    extents_uv = coords.max(axis=0) - coords.min(axis=0)
    return float(extents_uv[0] * extents_uv[1])

def _ensure_volume(mesh: trimesh.Trimesh, repair_mode: str = "light") -> trimesh.Trimesh:
    if mesh.is_volume:
        return mesh
    fixed = mesh.copy()
    try:
        fixed = repair_mesh(fixed, mode=repair_mode)
    except Exception:
        pass
    if not fixed.is_volume:
        try:
            fixed.process(validate=True)
        except Exception:
            return fixed
    return fixed

def _clean_mesh(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    cleaned = mesh.copy()
    try:
        cleaned.remove_infinite_values()
        cleaned.remove_unreferenced_vertices()
    except Exception:
        pass
    try:
        cleaned.process(validate=True)
    except Exception:
        pass
    return cleaned

def _finite_vertices(vertices: np.ndarray) -> np.ndarray:
    if vertices is None or len(vertices) == 0:
        return np.array([], dtype=float).reshape(0, 3)
    mask = np.isfinite(vertices).all(axis=1)
    return vertices[mask]


def _cap_patch_from_section(
    section_path: Optional[trimesh.path.Path3D],
    section_vertices: np.ndarray,
    plane_origin: np.ndarray,
    plane_normal: np.ndarray,
    depth: float,
    engine: Optional[str],
    direction: float,
) -> Optional[trimesh.Trimesh]:
    patches = []
    if section_path is not None:
        try:
            planar, to_3d = section_path.to_2D(normal=plane_normal)
            polygons = list(getattr(planar, "polygons_full", []))
            for poly in polygons:
                patch = trimesh.creation.extrude_polygon(poly, height=depth, engine=engine)
                if direction < 0:
                    patch.apply_translation([0.0, 0.0, -depth])
                patch.apply_transform(to_3d)
                patch = _clean_mesh(patch)
                patches.append(patch)
        except Exception:
            patches = []

    if not patches and section_vertices is not None and len(section_vertices) >= 3:
        try:
            from shapely.geometry import MultiPoint

            _, u, v = plane_basis(plane_normal)
            coords = np.vstack(
                (
                    np.dot(section_vertices - plane_origin, u),
                    np.dot(section_vertices - plane_origin, v),
                )
            ).T
            if np.isfinite(coords).all():
                poly = MultiPoint(coords).convex_hull
                if not poly.is_empty and poly.area > 0:
                    patch = trimesh.creation.extrude_polygon(
                        poly, height=depth, engine=engine
                    )
                    if direction < 0:
                        patch.apply_translation([0.0, 0.0, -depth])
                    transform = np.eye(4)
                    transform[:3, 0] = u
                    transform[:3, 1] = v
                    transform[:3, 2] = plane_normal / np.linalg.norm(plane_normal)
                    transform[:3, 3] = plane_origin
                    patch.apply_transform(transform)
                    patch = _clean_mesh(patch)
                    patches.append(patch)
        except Exception:
            pass

    if not patches:
        return None
    return trimesh.util.concatenate(patches)


def _remove_cap_faces(
    mesh: trimesh.Trimesh,
    plane_origin: np.ndarray,
    plane_normal: np.ndarray,
    tol: float,
) -> trimesh.Trimesh:
    if mesh.faces is None or len(mesh.faces) == 0:
        return mesh
    if not np.isfinite(mesh.vertices).all():
        return mesh
    if not np.isfinite(plane_origin).all() or not np.isfinite(plane_normal).all():
        return mesh
    norm = np.linalg.norm(plane_normal)
    if norm == 0.0:
        return mesh
    n = plane_normal / norm
    centers = mesh.triangles_center
    if not np.isfinite(centers).all():
        return mesh
    distances = np.abs((centers - plane_origin) @ n)
    face_normals = mesh.face_normals
    if not np.isfinite(face_normals).all():
        return mesh
    aligned = np.abs(face_normals @ n) > 0.9
    keep = ~(aligned & (distances < tol))
    if keep.all():
        return mesh
    try:
        return mesh.submesh([np.nonzero(keep)[0]], append=True)
    except Exception:
        return mesh

def apply_connectors(
    pos_mesh: trimesh.Trimesh,
    neg_mesh: trimesh.Trimesh,
    plane_origin: np.ndarray,
    plane_normal: np.ndarray,
    config: ConnectorConfig,
    engine: str,
    label: Optional[str] = None,
    ai_key: Optional[str] = None,
    repair_mode: str = "light",
    section_vertices: Optional[np.ndarray] = None,
    section_area: Optional[float] = None,
    section_path: Optional[trimesh.path.Path3D] = None,
) -> Tuple[trimesh.Trimesh, trimesh.Trimesh]:
    """
    Intelligently applies connectors and optional assembly labels.
    """
    base_pos = pos_mesh
    base_neg = neg_mesh

    # Heuristic for 'auto'
    vertices = section_vertices
    area = section_area
    if vertices is None or area is None:
        try:
            section = pos_mesh.section(plane_origin=plane_origin, plane_normal=plane_normal)
            if section is not None:
                vertices = section.vertices
                area = section.area
        except Exception:
            pass

    vertices = _finite_vertices(vertices)
    if area is None:
        area = 0.0

    if area <= 0.0:
        area = _section_bbox_area(vertices, plane_origin, plane_normal)

    style = config.style
    if style == "auto":
        # First: Check if thin-walled section -> use lip/groove
        if is_thin_walled_section(vertices, section_path, area, config.thin_wall_threshold):
            style = "lip"
            print("DEBUG: Detected thin-walled section, using lip & groove connectors")
        # AI Decision
        elif ai_key:
            from .ai import recommend_connector_with_gemini
            style = recommend_connector_with_gemini(vertices, ai_key)
            print(f"Gemini recommended connector: {style}")
        # Fallback Heuristic
        elif area <= 0:
            style = "hex"
        elif area > 800:
            style = "dovetail"
        elif area < 30:
            style = "none"
        else:
            style = "hex"

    if style == "none" and not label:
        return pos_mesh, neg_mesh

    # Positions for connectors
    conn_margin = config.peg_radius_mm * 2.5
    if style == "dovetail":
        conn_margin = config.dovetail_width_mm * 1.5
        
    positions = connector_positions(
        vertices, plane_origin, plane_normal, config.count, conn_margin
    )

    rot = rotation_from_z(plane_normal)
    pegs = []
    sockets = []
    
    # Handle Labeling
    if label:
        text_mesh = embossed_text_mesh(label, height=5.0) # 5mm tall letters
        # Move it to the center of the cut
        center = np.mean(vertices, axis=0) if len(vertices) > 0 else plane_origin
        # Orient it. Text mesh is in XY plane, pointing +Z. 
        # We want it on the cut face.
        text_mesh.apply_transform(rot)
        # Shift slightly so it's half-in, half-out or fully recessed.
        # Let's do 1mm recess.
        depth = 2.0 # Thickness of the box/letter
        text_mesh.apply_translation(center - plane_normal * (depth * 0.5))
        
        # We subtract from both sides to create a "shared" label cavity? 
        # Actually, let's subtract from neg and add/subtract from pos?
        # Better: Subtract from both to create a clear mark.
        if text_mesh.is_volume:
            sockets.append(text_mesh)
        # To avoid overlaps with pegs, we should check distance.
        # For now, just append.

    for i, pos in enumerate(positions):
        if style == "hex":
            r, d = config.peg_radius_mm, config.peg_depth_mm
            tol = config.tolerance_mm
            
            # Keying logic: scale the first peg slightly to enforce orientation
            if i == 0 and len(positions) > 1:
                r *= 1.2
            
            p = make_hex_prism(r, d)
            p.apply_transform(rot)
            p.apply_translation(pos + plane_normal * (d * 0.5))
            pegs.append(p)

            s = make_hex_prism(r + tol, d + tol * 2)
            s.apply_transform(rot)
            s.apply_translation(pos - plane_normal * (d * 0.5 + tol))
            sockets.append(s)

        elif style == "magnet":
            r, d = config.magnet_radius_mm, config.magnet_depth_mm
            # Magnets are just holes on both sides
            s1 = trimesh.creation.cylinder(radius=r, height=d*2)
            s1.apply_transform(rot)
            s1.apply_translation(pos)
            pegs.append(s1) # We use pegs list to indicate "subtract from pos" here? 
            # Actually let's just use subtraction for both if it's magnet holes.
            
            s2 = trimesh.creation.cylinder(radius=r, height=d*2)
            s2.apply_transform(rot)
            s2.apply_translation(pos)
            sockets.append(s2)

        elif style == "dovetail":
            w, h, d = config.dovetail_width_mm, config.peg_radius_mm * 2, config.peg_depth_mm
            tol = config.tolerance_mm
            
            p = make_dovetail(w, h, d, config.dovetail_angle_deg)
            # Dovetail points along Y in our maker, needs to point along normal
            # So we rotate it.
            p.apply_transform(trimesh.transformations.rotation_matrix(np.pi/2, [1,0,0]))
            p.apply_transform(rot)
            p.apply_translation(pos)
            pegs.append(p)

            s = make_dovetail(w + tol*2, h + tol*2, d + tol, config.dovetail_angle_deg)
            s.apply_transform(trimesh.transformations.rotation_matrix(np.pi/2, [1,0,0]))
            s.apply_transform(rot)
            s.apply_translation(pos)
            sockets.append(s)

    # Handle lip & groove style separately (not position-based)
    if style == "lip":
        lip_mesh, groove_mesh = make_lip_groove(
            section_path=section_path,
            section_vertices=vertices,
            plane_origin=plane_origin,
            plane_normal=plane_normal,
            lip_height=config.lip_height_mm,
            lip_width=config.lip_width_mm,
            tolerance=config.tolerance_mm,
            engine=engine,
        )
        
        if lip_mesh is not None and groove_mesh is not None:
            # Lip goes on positive side, groove on negative side
            # Position: lip protrudes from cut plane on pos side
            lip_mesh.apply_translation(plane_normal * 0.01)  # Tiny offset to ensure contact
            groove_mesh.apply_translation(-plane_normal * (config.lip_height_mm * 0.5))
            
            pegs = [lip_mesh]
            sockets = [groove_mesh]
            print("DEBUG: Lip & groove connectors created successfully")
        else:
            print("DEBUG: Lip & groove creation failed, falling back to no connectors")
            return pos_mesh, neg_mesh

    try:
        if not pegs:
            return pos_mesh, neg_mesh

        # Clean and AGGRESSIVELY REPAIR meshes before booleans
        pos_mesh = _clean_mesh(base_pos)
        neg_mesh = _clean_mesh(base_neg)
        
        # Apply aggressive repair using PyMeshFix (if available)
        print("DEBUG: Repairing meshes for boolean operations...")
        pos_mesh = repair_mesh(pos_mesh, mode=repair_mode)
        neg_mesh = repair_mesh(neg_mesh, mode=repair_mode)
        print(f"DEBUG: pos_mesh.is_volume={pos_mesh.is_volume}, neg_mesh.is_volume={neg_mesh.is_volume}")

        peg_comb = trimesh.util.concatenate(pegs)
        sock_comb = trimesh.util.concatenate(sockets)

        peg_comb = _ensure_volume(peg_comb, repair_mode=repair_mode)
        sock_comb = _ensure_volume(sock_comb, repair_mode=repair_mode)

        # Try multiple boolean engines
        engines_to_try = [engine, "blender", "scad"]
        
        pos_success = False
        neg_success = False
        
        # 1. Try to apply PEGS (Union to Positive Mesh)
        peg_errors = []
        for try_engine in engines_to_try:
            try:
                if style == "magnet":
                    # Magnets are holes on both sides
                    res = boolean_op([pos_mesh.copy(), peg_comb], op="difference", engine=try_engine)
                else:
                    res = boolean_op([pos_mesh.copy(), peg_comb], op="union", engine=try_engine)
                
                if res is not None and len(res.faces) > 0:
                    pos_mesh = res
                    pos_success = True
                    print(f"DEBUG: Peg boolean SUCCESS with engine={try_engine}")
                    break
            except Exception as e:
                peg_errors.append(f"{try_engine}: {e}")
                continue
        
        if not pos_success:
            print(f"DEBUG: All peg boolean engines failed: {peg_errors}")


        if not pos_success:
            print("Warning: Connector boolean failed. Attempting overlay fallback...")
            
            # --- OVERLAY FALLBACK LOGIC ---
            # Create solid patches to hold the connectors
            try:
                patch_depth = config.peg_depth_mm + config.tolerance_mm * 2
                if style == "dovetail":
                     patch_depth = config.dovetail_width_mm # Deeper for dovetails?
                
                # Make patches
                pos_patch = _cap_patch_from_section(
                    section_path, section_vertices, plane_origin, plane_normal, 
                    depth=patch_depth, engine=engine, direction=1.0
                )
                neg_patch = _cap_patch_from_section(
                    section_path, section_vertices, plane_origin, plane_normal, 
                    depth=patch_depth, engine=engine, direction=-1.0
                )
                
                if pos_patch is None or neg_patch is None:
                    # Fallback to box if section extraction failed
                    print("DEBUG: Section patch failed, utilizing bounding box patch")
                    area_ = _section_bbox_area(section_vertices, plane_origin, plane_normal)
                    if area_ > 0:
                        side = np.sqrt(area_) 
                        # This bbox logic is a bit weak, but better than nothing
                        # We really need the patch from section.
                        pass # relying on _cap_patch_from_section to handle it or return None

                if pos_patch and neg_patch:
                    # Apply Connectors to Patches
                    # POSITIVE: Union Pegs (or difference for magnets)
                    if style == "magnet":
                         res_p = boolean_op([pos_patch, peg_comb], op="difference", engine=engine, check_volume=False)
                    else:
                         res_p = boolean_op([pos_patch, peg_comb], op="union", engine=engine, check_volume=False)
                    
                    if res_p and len(res_p.faces) > 0:
                         pos_patch = res_p
                    
                    # NEGATIVE: Difference Sockets
                    if style == "magnet":
                        res_n = boolean_op([neg_patch, sock_comb], op="difference", engine=engine, check_volume=False)
                    else:
                        res_n = boolean_op([neg_patch, sock_comb], op="difference", engine=engine, check_volume=False)

                    if res_n and len(res_n.faces) > 0:
                        neg_patch = res_n
                        
                    # Concatenate to base meshes
                    final_pos = trimesh.util.concatenate([base_pos, pos_patch])
                    final_neg = trimesh.util.concatenate([base_neg, neg_patch])
                    
                    print("DEBUG: Overlay fallback successful. Connectors applied via patches.")
                    return final_pos, final_neg
            
            except Exception as e:
                print(f"Warning: Overlay fallback failed: {e}")

            print("Warning: Connector boolean failed. Skipping connectors for this cut.")
            return base_pos, base_neg

        # 2. Try to apply SOCKETS (Difference from Negative Mesh)
        for try_engine in engines_to_try:
            try:
                if style == "magnet":
                    res = boolean_op([neg_mesh.copy(), sock_comb], op="difference", engine=try_engine)
                else:
                    res = boolean_op([neg_mesh.copy(), sock_comb], op="difference", engine=try_engine)
                
                if res is not None and len(res.faces) > 0:
                    neg_mesh = res
                    neg_success = True
                    break
            except Exception:
                continue

        # Report status
        if pos_success and neg_success:
            return pos_mesh, neg_mesh
        elif pos_success:
            print("Warning: Sockets failed (boolean error), but Pegs were applied.")
            return pos_mesh, base_neg  # Return pos with connectors, neg without
        
        # If everything failed, return without connectors
        print(f"Warning: All connector attempts failed. Returning meshes without connectors.")
        return base_pos, base_neg
        
    except Exception as e:
        print(f"Warning: Connector generation failed ({e}), returning meshes without connectors.")
        return base_pos, base_neg

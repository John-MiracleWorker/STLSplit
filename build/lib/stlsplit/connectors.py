from __future__ import annotations

from typing import List, Tuple

import numpy as np
import trimesh

from .config import ConnectorConfig
from .mesh_ops import boolean_op, plane_basis, rotation_from_z


def connector_positions(
    section_vertices: np.ndarray,
    plane_origin: np.ndarray,
    plane_normal: np.ndarray,
    connector_count: int,
    margin_mm: float,
) -> List[np.ndarray]:
    """
    Calculate peg positions based on the 2D cross-section of the cut.
    """
    if section_vertices is None or len(section_vertices) == 0:
        return [plane_origin]

    _, u, v = plane_basis(plane_normal)
    
    # Project vertices onto the plane's 2D coordinate system
    coords = np.vstack(
        (
            np.dot(section_vertices - plane_origin, u),
            np.dot(section_vertices - plane_origin, v),
        )
    ).T
    
    min_uv = coords.min(axis=0)
    max_uv = coords.max(axis=0)
    extents_uv = max_uv - min_uv
    center_uv = (min_uv + max_uv) * 0.5

    # Choose the longer axis for spreading multiple connectors
    if extents_uv[0] >= extents_uv[1]:
        axis_idx = 0
        spread_axis = u
    else:
        axis_idx = 1
        spread_axis = v

    max_offset = (extents_uv[axis_idx] * 0.5) - margin_mm
    
    if connector_count <= 1 or max_offset <= 0:
        # Just one in the center
        pos = plane_origin + u * center_uv[0] + v * center_uv[1]
        return [pos]

    # Spread along the chosen axis
    offsets = np.linspace(-max_offset, max_offset, connector_count)
    base_pos = plane_origin + u * center_uv[0] + v * center_uv[1]
    
    # Remove the center component of the spread axis before adding offsets
    # Wait, it's easier to just do:
    results = []
    for off in offsets:
        p = base_pos + spread_axis * (off - (np.dot(base_pos - plane_origin, spread_axis)))
        # Simplified:
        p = plane_origin + u * center_uv[0] + v * center_uv[1]
        if axis_idx == 0:
            p = plane_origin + u * off + v * center_uv[1]
        else:
            p = plane_origin + u * center_uv[0] + v * off
        results.append(p)
    
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

    # Vertices for a wedge pointing along Y+
    # x - width, y - depth, z - height
    v = np.array([
        [-width/2, 0, -height/2], [width/2, 0, -height/2], [width/2, 0, height/2], [-width/2, 0, height/2],
        [-top_width/2, depth, -height/2], [top_width/2, depth, -height/2], [top_width/2, depth, height/2], [-top_width/2, depth, height/2]
    ])
    f = np.array([
        [0, 1, 2], [0, 2, 3], # Bottom
        [4, 6, 5], [4, 7, 6], # Top
        [0, 4, 5], [0, 5, 1], # Side 1
        [1, 5, 6], [1, 6, 2], # Side 2
        [2, 6, 7], [2, 7, 3], # Side 3
        [3, 7, 4], [3, 4, 0]  # Side 4
    ])
    return trimesh.Trimesh(vertices=v, faces=f)


from .mesh_ops import boolean_op, embossed_text_mesh

def apply_connectors(
    pos_mesh: trimesh.Trimesh,
    neg_mesh: trimesh.Trimesh,
    plane_origin: np.ndarray,
    plane_normal: np.ndarray,
    config: ConnectorConfig,
    engine: str,
    label: Optional[str] = None,
) -> Tuple[trimesh.Trimesh, trimesh.Trimesh]:
    """
    Intelligently applies connectors and optional assembly labels.
    """
    # Heuristic for 'auto'
    try:
        section = pos_mesh.section(plane_origin=plane_origin, plane_normal=plane_normal)
        if section is None:
            return pos_mesh, neg_mesh
        vertices = section.vertices
        area = section.area
        # Find a good spot for the label (usually the center of the largest polygon)
    except Exception:
        return pos_mesh, neg_mesh

    style = config.style
    if style == "auto":
        if area > 1000:
            style = "dovetail"
        elif area < 100:
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
        center = np.mean(vertices, axis=0)
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

    try:
        if not pegs:
            return pos_mesh, neg_mesh
            
        peg_comb = trimesh.util.concatenate(pegs)
        sock_comb = trimesh.util.concatenate(sockets)

        if style == "magnet":
            # Difference for both
            pos_mesh = boolean_op([pos_mesh, peg_comb], op="difference", engine=engine)
            neg_mesh = boolean_op([neg_mesh, sock_comb], op="difference", engine=engine)
        else:
            pos_mesh = boolean_op([pos_mesh, peg_comb], op="union", engine=engine)
            neg_mesh = boolean_op([neg_mesh, sock_comb], op="difference", engine=engine)
            
        return pos_mesh, neg_mesh
    except Exception as e:
        raise RuntimeError(f"Boolean failed for {style}: {e}")

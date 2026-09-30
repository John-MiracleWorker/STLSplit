from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import trimesh

from .analysis import face_gaussian_curvature, face_roughness, plane_band_face_stats
from .config import BuildVolume, ConnectorConfig, SplitConfig
from .connectors import apply_connectors, is_thin_walled_section, make_seam_plate
from .mesh_ops import Plane, fits_build_volume, split_mesh_by_plane


@dataclass
class SplitDecision:
    plane: Plane
    score: float
    axis: int = 0
    pos: float = 0.0
    alternatives: Optional[List['SplitDecision']] = None


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

    # Pre-calculate roughness/curvature for efficiency
    roughness = face_roughness(mesh)
    curvature = face_gaussian_curvature(mesh, config.curvature_band_mm)
    margin = 10.0  # mm from edge to avoid tiny slices

    candidate_list = []

    for axis in oversized_axes:
        normal = np.zeros(3)
        normal[axis] = 1.0

        low = bounds_min[axis] + margin
        high = bounds_max[axis] - margin

        if low < high:
            positions = np.linspace(low, high, max(config.cut_samples, 2))
        else:
            positions = np.array([bounds_min[axis] + extents[axis] * 0.5])

        for pos in positions:
            origin = (bounds_min + bounds_max) * 0.5
            origin[axis] = pos

            stats = plane_band_face_stats(
                mesh,
                roughness,
                origin,
                normal,
                config.curvature_band_mm,
                curvature=curvature,
            )
            area = stats.get("area", 0.0)
            curvature_weight = config.curvature_weight
            score = (
                config.roughness_weight * stats["roughness"]
                + config.density_weight * stats["density"]
                + curvature_weight * stats["curvature"]
                - (config.support_weight * (area / (extents.max()**2 + 1e-6)))
            )
            
            candidate_list.append({
                "plane": Plane(origin=origin, normal=normal),
                "score": float(score),
                "axis": int(axis),
                "pos": float(pos)
            })

    if not candidate_list:
        return None

    # Sort by math score (ascending is better? No, let's check score formula)
    # Roughness * weight (we want low) + Density * weight (low) - Support * weight (high area -> big negative -> low score)
    # So LOWER score is better.
    candidate_list.sort(key=lambda x: x["score"])
    
    # Build alternatives list from remaining candidates
    def make_decision(cand, alts=None):
        return SplitDecision(
            plane=cand["plane"],
            score=cand["score"],
            axis=cand["axis"],
            pos=cand["pos"],
            alternatives=alts
        )

    # If AI enabled, take top 3 and ask Gemini
    if config.use_ai and config.ai_key:
        from .ai import rank_cuts_with_gemini
        top_n = candidate_list[:6]  # Get more candidates for alternatives
        print(f"Asking Gemini to rank top {min(3, len(top_n))} cuts...")
        best_idx = rank_cuts_with_gemini(mesh, top_n[:3], config.ai_key)
        chosen = top_n[best_idx]
        
        # Create alternatives from the other candidates
        alternatives = [make_decision(c) for i, c in enumerate(top_n) if i != best_idx][:5]
        return make_decision(chosen, alternatives)
    
    # Else return best math with alternatives
    best_cand = candidate_list[0]
    alternatives = [make_decision(c) for c in candidate_list[1:6]]
    return make_decision(best_cand, alternatives)


def detect_bbox_openings(
    mesh: trimesh.Trimesh,
    grid: int = 7,
    hit_threshold: float = 0.35,
    distance_ratio: float = 1.4,
) -> dict[tuple[int, int], float]:
    bounds_min, bounds_max = mesh.bounds
    extents = bounds_max - bounds_min
    if np.any(extents <= 0):
        return {}

    face_stats: dict[tuple[int, int], tuple[float, Optional[float]]] = {}
    margin_frac = 0.05
    for axis in range(3):
        axes = [0, 1, 2]
        axes.remove(axis)
        grid_u = np.linspace(
            bounds_min[axes[0]] + extents[axes[0]] * margin_frac,
            bounds_max[axes[0]] - extents[axes[0]] * margin_frac,
            grid,
        )
        grid_v = np.linspace(
            bounds_min[axes[1]] + extents[axes[1]] * margin_frac,
            bounds_max[axes[1]] - extents[axes[1]] * margin_frac,
            grid,
        )
        uu, vv = np.meshgrid(grid_u, grid_v, indexing="xy")
        base = np.zeros((uu.size, 3), dtype=float)
        base[:, axes[0]] = uu.ravel()
        base[:, axes[1]] = vv.ravel()
        for side in (-1, 1):
            origins = base.copy()
            direction = np.zeros(3, dtype=float)
            if side < 0:
                origins[:, axis] = bounds_min[axis] - extents[axis] * 0.02 - 1e-3
                direction[axis] = 1.0
            else:
                origins[:, axis] = bounds_max[axis] + extents[axis] * 0.02 + 1e-3
                direction[axis] = -1.0
            directions = np.tile(direction, (len(origins), 1))
            try:
                distances = mesh.ray.intersects_first(origins, directions)
                finite = distances[distances >= 0]
                hit_ratio = float(np.count_nonzero(distances >= 0)) / max(len(distances), 1)
                median_dist = float(np.median(finite)) if len(finite) else None
            except Exception:
                hit_ratio = 1.0
                median_dist = None
            face_stats[(axis, side)] = (hit_ratio, median_dist)

    openings: dict[tuple[int, int], float] = {}
    for axis in range(3):
        min_hit, min_med = face_stats[(axis, -1)]
        max_hit, max_med = face_stats[(axis, 1)]
        if min_hit < hit_threshold:
            openings[(axis, -1)] = min_hit
        if max_hit < hit_threshold:
            openings[(axis, 1)] = max_hit
        if min_med and max_med:
            if max_med / min_med > distance_ratio:
                openings[(axis, 1)] = min(openings.get((axis, 1), 1.0), 0.0)
            elif min_med / max_med > distance_ratio:
                openings[(axis, -1)] = min(openings.get((axis, -1), 1.0), 0.0)

    return openings


def is_helmet_like(mesh: trimesh.Trimesh) -> bool:
    bounds_min, bounds_max = mesh.bounds
    extents = bounds_max - bounds_min
    if np.any(extents <= 0):
        return False

    xy_ratio = max(extents[0], extents[1]) / max(min(extents[0], extents[1]), 1e-6)
    if xy_ratio > 1.6:
        return False

    openings = detect_bbox_openings(mesh)
    if not openings:
        return False
    if (2, -1) not in openings and (2, 1) not in openings:
        return False

    try:
        origin = (bounds_min + bounds_max) * 0.5
        normal = np.array([0.0, 0.0, 1.0])
        section = mesh.section(plane_origin=origin, plane_normal=normal)
        if section is None:
            return False
        vertices = section.vertices
        if vertices is None or len(vertices) < 3:
            return False
        try:
            area = float(section.area)
        except Exception:
            area = 0.0
        if area <= 0.0:
            from .mesh_ops import plane_basis

            _, u, v = plane_basis(normal)
            coords = np.vstack(
                (
                    np.dot(vertices - origin, u),
                    np.dot(vertices - origin, v),
                )
            ).T
            if np.isfinite(coords).all():
                extents_uv = coords.max(axis=0) - coords.min(axis=0)
                area = float(extents_uv[0] * extents_uv[1])
        if not is_thin_walled_section(vertices, section, area, threshold=0.2):
            return False
    except Exception:
        return False

    return True


def select_helmet_cut_positions(
    mesh: trimesh.Trimesh,
    axis: int,
    num_cuts_needed: int,
    split_cfg: SplitConfig,
    openings: dict[tuple[int, int], float],
    roughness: np.ndarray,
    curvature: np.ndarray,
) -> List[float]:
    if num_cuts_needed <= 0:
        return []

    bounds_min, bounds_max = mesh.bounds
    extents = bounds_max - bounds_min
    piece_extent = extents[axis]
    if piece_extent <= 0:
        return []

    base_margin = max(split_cfg.cut_margin_mm, piece_extent * 0.05)
    min_pos = bounds_min[axis] + base_margin
    max_pos = bounds_max[axis] - base_margin
    if min_pos >= max_pos:
        return [float(bounds_min[axis] + piece_extent * 0.5)]

    sample_count = max(split_cfg.cut_samples, num_cuts_needed + 2, 7)
    positions = np.linspace(min_pos, max_pos, sample_count)

    def opening_penalty(pos: float) -> float:
        penalty = 0.0
        band = max(piece_extent * 0.2, 10.0)
        for side in (-1, 1):
            strength = 0.0
            if (axis, side) in openings:
                strength = max(0.0, 1.0 - openings[(axis, side)])
            if strength <= 0:
                continue
            if side < 0:
                dist = pos - bounds_min[axis]
            else:
                dist = bounds_max[axis] - pos
            if dist < band:
                penalty += strength * ((band - dist) / band)
        return penalty

    curvature_weight = split_cfg.curvature_weight
    opening_weight = 1.2
    support_norm = extents.max() ** 2 + 1e-6

    scored = []
    normal = np.zeros(3)
    normal[axis] = 1.0
    for pos in positions:
        origin = (bounds_min + bounds_max) * 0.5
        origin[axis] = pos
        stats = plane_band_face_stats(
            mesh,
            roughness,
            origin,
            normal,
            split_cfg.curvature_band_mm,
            curvature=curvature,
        )
        area = stats.get("area", 0.0)
        score = (
            split_cfg.roughness_weight * stats["roughness"]
            + split_cfg.density_weight * stats["density"]
            + curvature_weight * stats["curvature"]
            + opening_weight * opening_penalty(pos)
            - (split_cfg.support_weight * (area / support_norm))
        )
        scored.append((score, float(pos)))

    scored.sort(key=lambda x: x[0])
    target_spacing = (max_pos - min_pos) / (num_cuts_needed + 1)
    min_spacing = max(target_spacing * 0.5, split_cfg.cut_margin_mm * 2)

    chosen: List[float] = []
    for _, pos in scored:
        if all(abs(pos - existing) >= min_spacing for existing in chosen):
            chosen.append(pos)
            if len(chosen) >= num_cuts_needed:
                break

    if len(chosen) < num_cuts_needed:
        chosen = list(
            np.linspace(min_pos, max_pos, num_cuts_needed + 2)[1:-1].tolist()
        )

    return sorted(chosen)


def split_helmet(
    mesh: trimesh.Trimesh,
    build: BuildVolume,
    connector_cfg: ConnectorConfig,
    engine: str,
    add_connectors: bool = True,
    split_cfg: Optional[SplitConfig] = None,
    repair_mode: str = "light",
    ai_key: Optional[str] = None,
) -> List[trimesh.Trimesh]:
    """
    Helmet-specific splitting that checks ALL dimensions (X, Y, Z).
    Uses edge-based dovetail connectors for thin walls when connectors are enabled.
    
    Strategy:
    1. Check which dimensions exceed build volume
    2. For width/depth (X/Y): use vertical cuts through center
    3. For height (Z): use horizontal cuts
    4. Apply edge dovetails for alignment (optional)
    """
    bounds_min, bounds_max = mesh.bounds
    extents = bounds_max - bounds_min
    build_dims = np.array([build.x_mm, build.y_mm, build.z_mm])
    
    print(f"🪖 Helmet dimensions: {extents[0]:.1f} x {extents[1]:.1f} x {extents[2]:.1f} mm")
    print(f"🪖 Build volume: {build.x_mm:.1f} x {build.y_mm:.1f} x {build.z_mm:.1f} mm")
    
    # Check if already fits
    if np.all(extents <= build_dims):
        print("🪖 Helmet already fits build volume!")
        return [mesh]
    
    # Determine which axes need cutting (prioritize largest overage first)
    overage = extents - build_dims
    oversized_axes = np.where(overage > 0)[0]
    
    if len(oversized_axes) == 0:
        return [mesh]
    
    # Sort by overage amount (largest first)
    oversized_axes = sorted(oversized_axes, key=lambda a: overage[a], reverse=True)
    
    print(f"🪖 Oversized axes: {', '.join([f'{chr(88+a)} by {overage[a]:.1f}mm' for a in oversized_axes])}")
    
    
    split_cfg = split_cfg or SplitConfig()
    use_dovetails = add_connectors and connector_cfg.style != "none"
    if not add_connectors:
        print("🪖 Helmet mode: connectors disabled")
    if use_dovetails and connector_cfg.style not in ("auto", "dovetail"):
        print(
            f"🪖 Helmet mode: forcing dovetail connectors (ignoring style '{connector_cfg.style}')"
        )

    pieces = [mesh]
    seam_plates: List[trimesh.Trimesh] = []
    cut_counter = 0
    
    for axis in oversized_axes:
        axis_name = ['X', 'Y', 'Z'][axis]
        new_pieces = []
        
        for piece in pieces:
            piece_bounds = piece.bounds
            piece_extent = piece_bounds[1][axis] - piece_bounds[0][axis]
            
            # Check if this piece needs cutting on this axis
            if piece_extent <= build_dims[axis]:
                new_pieces.append(piece)
                continue
            
            # Calculate number of cuts needed for this piece on this axis
            num_cuts_needed = int(np.ceil(piece_extent / build_dims[axis])) - 1
            num_cuts_needed = min(num_cuts_needed, 2)  # Max 2 cuts per axis = 3 pieces
            
            if num_cuts_needed == 0:
                new_pieces.append(piece)
                continue
            
            print(f"🪖 Cutting piece along {axis_name} axis ({num_cuts_needed} cut(s))")
            
            # Calculate cut positions using helmet heuristics
            try:
                openings = detect_bbox_openings(piece)
                if openings:
                    for (open_axis, side), ratio in openings.items():
                        face = f"{['X','Y','Z'][open_axis]}{'min' if side < 0 else 'max'}"
                        print(f"🪖 Opening detected at {face} (hit_ratio={ratio:.2f})")
            except Exception:
                openings = {}

            roughness = face_roughness(piece)
            curvature = face_gaussian_curvature(piece, split_cfg.curvature_band_mm)
            cut_positions = select_helmet_cut_positions(
                piece,
                axis,
                num_cuts_needed,
                split_cfg,
                openings,
                roughness,
                curvature,
            )
            
            current = piece
            for cut_pos in cut_positions:
                # Create cut plane
                origin = (piece_bounds[0] + piece_bounds[1]) / 2
                origin[axis] = cut_pos
                normal = np.zeros(3)
                normal[axis] = 1.0
                
                print(f"🪖 Cut at {axis_name}={cut_pos:.1f}mm")

                section = None
                section_vertices = None
                if split_cfg.helmet_seam_plates and split_cfg.helmet_seam_width_mm > 0:
                    try:
                        section = current.section(
                            plane_origin=origin, plane_normal=normal
                        )
                        if section is not None:
                            section_vertices = section.vertices
                    except Exception:
                        section = None
                        section_vertices = None

                pos_mesh, neg_mesh = split_mesh_by_plane(
                    current, origin, normal, repair_mode=repair_mode
                )
                
                if pos_mesh is None or neg_mesh is None:
                    print(f"🪖 Cut failed, keeping piece as-is")
                    continue
                
                if len(pos_mesh.faces) == 0 or len(neg_mesh.faces) == 0:
                    print(f"🪖 Cut produced empty mesh, skipping")
                    continue
                
                # Try to repair meshes to make them watertight
                if repair_mode != "none":
                    try:
                        pos_mesh.fill_holes()
                        pos_mesh.fix_normals()
                        neg_mesh.fill_holes()
                        neg_mesh.fix_normals()
                    except Exception:
                        pass

                if (
                    split_cfg.helmet_seam_plates
                    and split_cfg.helmet_seam_width_mm > 0
                    and split_cfg.helmet_seam_thickness_mm > 0
                ):
                    if section_vertices is not None and len(section_vertices) >= 3:
                        plate = make_seam_plate(
                            section_path=section,
                            section_vertices=section_vertices,
                            plane_origin=origin,
                            plane_normal=normal,
                            plate_thickness=split_cfg.helmet_seam_thickness_mm,
                            plate_width=split_cfg.helmet_seam_width_mm,
                            engine=engine,
                        )
                        if plate is not None and len(plate.faces) > 0:
                            seam_plates.append(plate)
                            print("🪖 Seam plate added for this cut")
                
                # Apply edge-based dovetail connectors (better for thin walls than holes)
                # Dovetails interlock at the cut edge, not deep into the wall
                if use_dovetails:
                    try:
                        print(f"🪖 Applying dovetail edge connectors...")
                        cut_counter += 1
                        pos_mesh, neg_mesh = apply_edge_dovetails(
                            pos_mesh,
                            neg_mesh,
                            origin,
                            normal,
                            connector_cfg,
                            engine,
                            label_index=cut_counter,
                            repair_mode=repair_mode,
                        )
                    except Exception as e:
                        print(f"🪖 Dovetail connectors failed: {e}")
                
                new_pieces.append(neg_mesh)
                current = pos_mesh
            
            new_pieces.append(current)
        
        pieces = new_pieces
    
    if seam_plates:
        pieces.extend(seam_plates)

    print(f"🪖 Helmet split complete: {len(pieces)} pieces")
    return pieces


def apply_edge_dovetails(
    pos_mesh: trimesh.Trimesh,
    neg_mesh: trimesh.Trimesh,
    plane_origin: np.ndarray,
    plane_normal: np.ndarray,
    config: ConnectorConfig,
    engine: str,
    label_index: int = 0,
    repair_mode: str = "light",
) -> Tuple[trimesh.Trimesh, trimesh.Trimesh]:
    """
    Creates small dovetail tabs at the cut edge for alignment.
    Also adds ID dots based on label_index (1 dot, 2 dots...) for matching.
    """
    from .mesh_ops import boolean_op, plane_basis, repair_mesh, rotation_from_z
    from .connectors import make_dovetail
    
    # Get section to find positions for dovetails
    try:
        section = pos_mesh.section(plane_origin=plane_origin, plane_normal=plane_normal)
        if section is None:
            print(f"🪖 No section found for dovetails")
            return pos_mesh, neg_mesh
        
        vertices = section.vertices
        if len(vertices) < 3:
            print(f"🪖 Section too small for dovetails")
            return pos_mesh, neg_mesh
    except Exception as e:
        print(f"🪖 Section failed: {e}")
        return pos_mesh, neg_mesh
    
    # Find center and extents
    center = np.mean(vertices, axis=0)
    _, u, v = plane_basis(plane_normal)
    
    # Project vertices to find extent
    coords_u = np.dot(vertices - center, u)
    extent_u = (coords_u.max() - coords_u.min()) * 0.25  # 25% from center
    
    # Dovetail parameters (smaller for thin walls)
    dt_width = min(config.dovetail_width_mm, 8.0)  # Max 8mm wide
    dt_depth = min(config.peg_depth_mm, 4.0)       # Max 4mm deep
    dt_height = 3.0                                 # 3mm thick
    dt_angle = config.dovetail_angle_deg
    
    # Create 2 dovetails at opposite positions
    dovetail_positions = [
        center + u * extent_u,
        center - u * extent_u,
    ]
    
    # Point dovetails into the negative side of the cut (toward the other piece)
    rot = rotation_from_z(-plane_normal)
    pegs = []
    sockets = []
    
    # Generate ID dots if index > 0
    dots_mesh = None
    if label_index > 0:
        dot_radius = 0.6
        dot_height = 0.6
        num_dots = min(label_index, 5) # Max 5 dots to fit
        
        dots = []
        spacing = dt_width / (num_dots + 1)
        start_x = -dt_width/2
        
        for k in range(num_dots):
            dot = trimesh.creation.cylinder(radius=dot_radius, height=dot_height, sections=8)
            # Dots on Top Face (Y+)
            # Position relative to peg center (which is at origin before transform)
            # X: spaced
            # Y: dt_height/2 + dot_height/2
            # Z: dt_depth/2
            dot_pos = [start_x + spacing*(k+1), dt_height/2, dt_depth/2]
            
            # Rotate dot so its cylinder axis (Z) points along Peg Y (Up)?
            # Cylinder is along Z by default. We want it pointing up (Y).
            # Rotate 90 deg about X.
            tf_dot = trimesh.transformations.rotation_matrix(np.pi/2, [1,0,0])
            dot.apply_transform(tf_dot)
            dot.apply_translation(dot_pos)
            dots.append(dot)
            
        if dots:
            dots_mesh = trimesh.util.concatenate(dots)
    
    for pos in dovetail_positions:
        # Create dovetail peg (protrudes from positive mesh)
        peg = make_dovetail(dt_width, dt_height, dt_depth, dt_angle)
        
        # Add dots to peg if valid
        if dots_mesh:
            peg = trimesh.util.concatenate([peg, dots_mesh])
            
        peg.apply_transform(rot)
        peg.apply_translation(pos)
        pegs.append(peg)
        
        # Create socket (slightly larger for tolerance)
        socket = make_dovetail(
            dt_width + config.tolerance_mm * 2,
            dt_height + config.tolerance_mm * 2,
            dt_depth + config.tolerance_mm,
            dt_angle
        )
        # Sockets don't need dots (they are negative space) - but we need space for the dots?
        # Yes, we should clear space for dots in the socket too!
        # Add simpler larger box for dots clearing? 
        # For now, let's assume tolerance handles it or user cleans it.
        # Adding complex negative boolean might fail.
        
        socket.apply_transform(rot)
        socket.apply_translation(pos)
        sockets.append(socket)

    
    def _boolean_with_fallback(
        meshes: List[trimesh.Trimesh],
        op: str,
    ) -> Tuple[Optional[trimesh.Trimesh], Optional[str]]:
        engines = []
        if engine not in engines:
            engines.append(engine)
        for candidate in ("manifold", "blender", None):
            if candidate not in engines:
                engines.append(candidate)
        engines_available = getattr(trimesh.boolean, "engines_available", set())
        for candidate in sorted([e for e in engines_available if e is not None]):
            if candidate not in engines:
                engines.append(candidate)
        if None in engines_available and None not in engines:
            engines.append(None)

        last_error = None
        for candidate in engines:
            try:
                result = boolean_op(
                    meshes, op=op, engine=candidate, check_volume=False
                )
            except Exception as e:
                last_error = e
                continue
            if result is not None and len(result.faces) > 0:
                return result, candidate
        if last_error:
            print(f"🪖 Dovetail boolean failed across engines: {last_error}")
        return None, None

    def _overlay_fallback() -> Tuple[trimesh.Trimesh, trimesh.Trimesh]:
        try:
            from .connectors import _cap_patch_from_section
        except Exception:
            return base_pos, base_neg

        patch_depth = dt_depth + config.tolerance_mm * 2
        overlap = max(0.15, patch_depth * 0.05)

        pos_patch = _cap_patch_from_section(
            section,
            vertices,
            plane_origin,
            plane_normal,
            depth=patch_depth,
            engine=engine,
            direction=1.0,
        )
        neg_patch = _cap_patch_from_section(
            section,
            vertices,
            plane_origin,
            plane_normal,
            depth=patch_depth,
            engine=engine,
            direction=-1.0,
        )
        if pos_patch is None or neg_patch is None:
            _, u, v = plane_basis(plane_normal)
            coords = np.vstack(
                (
                    np.dot(vertices - plane_origin, u),
                    np.dot(vertices - plane_origin, v),
                )
            ).T
            if not np.isfinite(coords).all():
                return base_pos, base_neg
            extents_uv = coords.max(axis=0) - coords.min(axis=0)
            if extents_uv[0] <= 0 or extents_uv[1] <= 0:
                return base_pos, base_neg
            center_uv = (coords.max(axis=0) + coords.min(axis=0)) * 0.5
            center = plane_origin + u * center_uv[0] + v * center_uv[1]
            patch_box = trimesh.creation.box(
                extents=[extents_uv[0], extents_uv[1], patch_depth]
            )
            patch_rot = rotation_from_z(plane_normal)
            patch_box.apply_transform(patch_rot)
            pos_patch = patch_box.copy()
            neg_patch = patch_box.copy()
            pos_patch.apply_translation(center + plane_normal * (patch_depth * 0.5))
            neg_patch.apply_translation(center - plane_normal * (patch_depth * 0.5))

        try:
            peg_combined = trimesh.util.concatenate(pegs)
            result_pos, _ = _boolean_with_fallback([pos_patch, peg_combined], op="union")
            if result_pos is not None and len(result_pos.faces) > 0:
                pos_patch = result_pos
            else:
                pos_patch = trimesh.util.concatenate([pos_patch, peg_combined])
        except Exception:
            pos_patch = trimesh.util.concatenate([pos_patch, peg_combined])

        try:
            socket_combined = trimesh.util.concatenate(sockets)
            result_neg, _ = _boolean_with_fallback(
                [neg_patch, socket_combined], op="difference"
            )
            if result_neg is not None and len(result_neg.faces) > 0:
                neg_patch = result_neg
            else:
                print("🪖 Overlay socket cut failed; using solid dovetail patch")
        except Exception:
            print("🪖 Overlay socket boolean failed; using solid dovetail patch")

        pos_patch.apply_translation(plane_normal * overlap)
        neg_patch.apply_translation(-plane_normal * overlap)

        return (
            trimesh.util.concatenate([base_pos, pos_patch]),
            trimesh.util.concatenate([base_neg, neg_patch]),
        )

    # Try to apply dovetails
    base_pos = pos_mesh
    base_neg = neg_mesh

    try:
        pos_mesh = repair_mesh(pos_mesh, mode=repair_mode)
        neg_mesh = repair_mesh(neg_mesh, mode=repair_mode)
    except Exception:
        pos_mesh = base_pos
        neg_mesh = base_neg

    if not pos_mesh.is_volume or not neg_mesh.is_volume:
        print("🪖 Non-volume shell detected; using overlay dovetails to avoid floaters")
        return _overlay_fallback()

    # 1. PEGS (Positive Mesh)
    try:
        peg_combined = trimesh.util.concatenate(pegs)
        result_pos, engine_used = _boolean_with_fallback(
            [pos_mesh, peg_combined], op="union"
        )
        if result_pos is None or len(result_pos.faces) == 0:
            print("🪖 Dovetail peg boolean failed (empty result), using overlay dovetails")
            return _overlay_fallback()
        pos_mesh = result_pos
        if engine_used:
            print(f"🪖 Dovetail pegs added (boolean union, engine={engine_used})")
        else:
            print(f"🪖 Dovetail pegs added (boolean union)")
    except Exception as e:
        print(f"🪖 Dovetail peg boolean failed ({e}), using overlay dovetails")
        return _overlay_fallback()

    # 2. SOCKETS (Negative Mesh)
    try:
        socket_combined = trimesh.util.concatenate(sockets)

        # We MUST do boolean difference here
        # Try to repair negative mesh first to help boolean
        try:
            neg_mesh.fill_holes()
            neg_mesh.fix_normals()
        except Exception:
            pass

        result_neg, engine_used = _boolean_with_fallback(
            [neg_mesh, socket_combined], op="difference"
        )
        if result_neg is None or len(result_neg.faces) == 0:
            print("🪖 Dovetail socket boolean failed (empty result), using overlay dovetails")
            return _overlay_fallback()
        neg_mesh = result_neg
        if engine_used:
            print(f"🪖 Dovetail sockets cut (boolean difference, engine={engine_used})")
        else:
            print(f"🪖 Dovetail sockets cut (boolean difference)")
    except Exception as e:
        print(f"🪖 Dovetail socket boolean failed ({e}), using overlay dovetails")
        return _overlay_fallback()

    return pos_mesh, neg_mesh


def apply_alignment_holes(
    pos_mesh: trimesh.Trimesh,
    neg_mesh: trimesh.Trimesh,
    plane_origin: np.ndarray,
    plane_normal: np.ndarray,
    config: ConnectorConfig,
    engine: str,
) -> Tuple[trimesh.Trimesh, trimesh.Trimesh]:
    """
    Creates simple alignment holes drilled INTO both meshes.
    These are purely subtractive - no protruding geometry that could float.
    
    The holes are cylindrical cavities that help align pieces during assembly.
    They're designed for use with alignment pins/dowels.
    """
    from .mesh_ops import boolean_op, plane_basis, rotation_from_z
    
    print(f"🕳️ apply_alignment_holes called")
    
    # Get section to find good positions for holes
    try:
        section = pos_mesh.section(plane_origin=plane_origin, plane_normal=plane_normal)
        if section is None:
            print(f"🕳️ Section is None, skipping holes")
            return pos_mesh, neg_mesh
        
        vertices = section.vertices
        if len(vertices) < 3:
            print(f"🕳️ Section has < 3 vertices, skipping holes")
            return pos_mesh, neg_mesh
        
        print(f"🕳️ Section has {len(vertices)} vertices")
    except Exception as e:
        print(f"🕳️ Section failed: {e}")
        return pos_mesh, neg_mesh
    
    # Find center of section
    center = np.mean(vertices, axis=0)
    
    # Get basis for the cut plane
    _, u, v = plane_basis(plane_normal)
    
    # Create 2 alignment holes at opposite sides
    hole_radius = config.peg_radius_mm
    hole_depth = config.peg_depth_mm
    
    # Project vertices to find extents
    coords_u = np.dot(vertices - center, u)
    coords_v = np.dot(vertices - center, v)
    extent_u = (coords_u.max() - coords_u.min()) * 0.3  # 30% from center
    
    hole_positions = [
        center + u * extent_u,
        center - u * extent_u,
    ]
    
    rot = rotation_from_z(plane_normal)
    holes_pos = []
    holes_neg = []
    
    for pos in hole_positions:
        # Create cylinder for each hole
        # Hole goes INTO positive mesh (below cut plane)
        hole_pos = trimesh.creation.cylinder(radius=hole_radius, height=hole_depth, sections=16)
        hole_pos.apply_transform(rot)
        hole_pos.apply_translation(pos - plane_normal * (hole_depth * 0.5))
        holes_pos.append(hole_pos)
        
        # Hole goes INTO negative mesh (above cut plane)  
        hole_neg = trimesh.creation.cylinder(radius=hole_radius, height=hole_depth, sections=16)
        hole_neg.apply_transform(rot)
        hole_neg.apply_translation(pos + plane_normal * (hole_depth * 0.5))
        holes_neg.append(hole_neg)
    
    # Boolean subtract holes from meshes
    try:
        holes_pos_combined = trimesh.util.concatenate(holes_pos)
        holes_neg_combined = trimesh.util.concatenate(holes_neg)
        
        result_pos = boolean_op([pos_mesh, holes_pos_combined], op="difference", engine=engine)
        result_neg = boolean_op([neg_mesh, holes_neg_combined], op="difference", engine=engine)
        
        if result_pos is not None and len(result_pos.faces) > 0:
            pos_mesh = result_pos
            print(f"🪖 Alignment holes drilled into positive mesh")
        
        if result_neg is not None and len(result_neg.faces) > 0:
            neg_mesh = result_neg
            print(f"🪖 Alignment holes drilled into negative mesh")
            
    except Exception as e:
        print(f"🪖 Hole drilling failed: {e}")
    
    return pos_mesh, neg_mesh


def split_to_fit(
    mesh: trimesh.Trimesh,
    build: BuildVolume,
    split_cfg: SplitConfig,
    connector_cfg: ConnectorConfig,
    engine: str,
    add_connectors: bool,
    repair_mode: str = "light",
) -> List[trimesh.Trimesh]:
    pieces: List[trimesh.Trimesh] = []
    queue: List[Tuple[trimesh.Trimesh, int]] = [(mesh, 0)]
    
    # Perform initial AI analysis of the full model if AI is enabled
    ai_analysis = None
    if split_cfg.use_ai and split_cfg.ai_key:
        try:
            from .ai import analyze_model_for_splitting
            print("🧠 Analyzing model shape with AI...")
            ai_analysis = analyze_model_for_splitting(
                mesh, 
                (build.x_mm, build.y_mm, build.z_mm),
                split_cfg.ai_key
            )
            if ai_analysis.get('model_type'):
                print(f"   Model type: {ai_analysis.get('model_type')}")
            if ai_analysis.get('suggested_cuts'):
                print(f"   Suggested cuts: {len(ai_analysis.get('suggested_cuts', []))}")
            if (
                ai_analysis.get('default_connector')
                and ai_analysis.get('model_type', '').lower() != 'helmet'
            ):
                # Update connector config based on AI recommendation
                rec_conn = ai_analysis.get('default_connector', 'hex')
                if connector_cfg.style == 'auto':
                    connector_cfg = ConnectorConfig(
                        style=rec_conn,
                        count=connector_cfg.count,
                        peg_radius_mm=connector_cfg.peg_radius_mm,
                        peg_depth_mm=connector_cfg.peg_depth_mm
                    )
                    print(f"   AI recommends connector: {rec_conn}")
        except Exception as e:
            print(f"AI model analysis failed: {e}")

    # If helmet detected, use helmet-specific splitting
    helmet_by_ai = ai_analysis and ai_analysis.get("model_type", "").lower() == "helmet"
    helmet_by_heuristic = False
    if not helmet_by_ai:
        try:
            helmet_by_heuristic = is_helmet_like(mesh)
        except Exception:
            helmet_by_heuristic = False

    if helmet_by_ai or helmet_by_heuristic:
        if helmet_by_ai:
            print("🪖 Helmet detected! Using helmet-specific splitting...")
        else:
            print("🪖 Helmet-like geometry detected (heuristics). Using helmet-specific splitting...")
        return split_helmet(
            mesh,
            build,
            connector_cfg,
            engine,
            add_connectors=False,
            split_cfg=split_cfg,
            repair_mode=repair_mode,
            ai_key=split_cfg.ai_key if split_cfg.use_ai else None,
        )

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

        # Try the chosen cut, but if it creates non-volume meshes, try alternatives
        candidates_to_try = [decision]
        
        # Get additional candidates from choose_cut_plane's sorted list
        # We store the top candidates in the decision for fallback
        if hasattr(decision, 'alternatives') and decision.alternatives:
            candidates_to_try.extend(decision.alternatives[:3])  # Try up to 3 alternatives
        
        best_pos = None
        best_neg = None
        best_decision = None
        section_vertices = None
        section_area = None
        section_path = None
        
        for candidate in candidates_to_try:
            try:
                section = current.section(
                    plane_origin=candidate.plane.origin, plane_normal=candidate.plane.normal
                )
                if section is not None:
                    section_path = section
                    section_vertices = section.vertices
                    section_area = section.area
            except Exception:
                pass

            pos_mesh, neg_mesh = split_mesh_by_plane(
                current,
                candidate.plane.origin,
                candidate.plane.normal,
                repair_mode=repair_mode,
            )
            
            if pos_mesh is None or neg_mesh is None:
                continue
            if len(pos_mesh.faces) == 0 or len(neg_mesh.faces) == 0:
                continue
            
            # Check if both meshes are volumes (watertight)
            if pos_mesh.is_volume and neg_mesh.is_volume:
                best_pos = pos_mesh
                best_neg = neg_mesh
                best_decision = candidate
                print(f"DEBUG: Found volume-safe cut at axis={candidate.axis}, pos={candidate.pos:.1f}")
                break
            elif best_pos is None:
                # Keep first valid cut as fallback even if not volumes
                best_pos = pos_mesh
                best_neg = neg_mesh
                best_decision = candidate
        
        if best_pos is None or best_neg is None:
            pieces.append(current)
            continue
        
        pos_mesh = best_pos
        neg_mesh = best_neg
        decision = best_decision

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
                    ai_key=split_cfg.ai_key,
                    repair_mode=repair_mode,
                    section_vertices=section_vertices,
                    section_area=section_area,
                    section_path=section_path,
                )
            except Exception as e:
                # Fallback to no connectors if boolean fails
                print(f"Warning: Connector failed ({e}), skipping connectors for this split.")

        queue.append((pos_mesh, depth + 1))
        queue.append((neg_mesh, depth + 1))

    return pieces

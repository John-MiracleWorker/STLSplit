from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import trimesh

from .analysis import face_roughness, plane_band_face_stats
from .config import BuildVolume, ConnectorConfig, SplitConfig
from .connectors import apply_connectors
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

    # Pre-calculate roughness for efficiency
    roughness = face_roughness(mesh)
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
                mesh, roughness, origin, normal, config.curvature_band_mm
            )
            area = stats.get("area", 0.0)
            
            score = (
                config.roughness_weight * stats["roughness"]
                + config.density_weight * stats["density"]
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
            if ai_analysis.get('default_connector'):
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

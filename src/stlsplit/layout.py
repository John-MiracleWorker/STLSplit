import numpy as np
import trimesh
from typing import List, Tuple
from .config import BuildVolume
from .mesh_ops import rotation_from_z

def orient_to_bed(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """
    Rotates the mesh so its largest planar face points down (-Z).
    This is optimal for printing split parts.
    """
    try:
        # Get facets (groups of coplanar adjacent faces)
        # Handle both old and new trimesh API
        facets = getattr(mesh, 'facets', None)
        if facets is None or len(facets) == 0:
            # Fallback: just put the largest bounding box face down
            extents = mesh.extents
            # Find which axis has the smallest extent (that's likely the "thin" direction)
            min_axis = np.argmin(extents)
            if min_axis != 2:
                # Rotate so that axis becomes Z
                if min_axis == 0:
                    matrix = trimesh.transformations.rotation_matrix(np.pi/2, [0, 1, 0])
                else:
                    matrix = trimesh.transformations.rotation_matrix(np.pi/2, [1, 0, 0])
                mesh = mesh.copy()
                mesh.apply_transform(matrix)
            min_z = mesh.bounds[0][2]
            mesh.apply_translation([0, 0, -min_z])
            return mesh

        # Calculate area for each facet
        facet_areas = []
        for facet_idxs in facets:
            facet_areas.append(np.sum(mesh.area_faces[facet_idxs]))
        
        best_idx = np.argmax(facet_areas)
        
        # Get normal of the largest facet
        facet_normals = getattr(mesh, 'facets_normal', None)
        if facet_normals is None:
            # Calculate normal manually from average of face normals
            best_normal = np.mean(mesh.face_normals[facets[best_idx]], axis=0)
            best_normal = best_normal / np.linalg.norm(best_normal)
        else:
            best_normal = facet_normals[best_idx]
        
        # We want this normal to point DOWN (-Z)
        target = np.array([0.0, 0.0, -1.0])
        
        # Align the facet normal to point down
        matrix = trimesh.geometry.align_vectors(best_normal, target)
        
        mesh = mesh.copy()
        mesh.apply_transform(matrix)
        
        # Also move min Z to 0
        min_z = mesh.bounds[0][2]
        mesh.apply_translation([0, 0, -min_z])
        return mesh

    except Exception as e:
        print(f"Warning: Orientation failed: {e}")
        return mesh

def arrange_on_plates(meshes: List[trimesh.Trimesh], build: BuildVolume) -> List[trimesh.Trimesh]:
    """
    Arranges meshes onto virtual plates.
    Rotates them flat first, then uses 2D bin packing.
    """
    arranged = []
    
    # Pre-orient all meshes
    oriented_meshes = [orient_to_bed(m) for m in meshes]
    
    # Sort by footprint area (largest first)
    oriented_meshes.sort(key=lambda m: m.extents[0] * m.extents[1], reverse=True)
    
    plate_width = build.x_mm
    plate_depth = build.y_mm
    plate_margin = 10.0 # mm spacing
    
    plate_offset_x = 0.0
    
    # Really simple shelf packing
    # We maintain a list of 'shelves' for the current plate
    # But to keep it robust and simple for now:
    # Just fill rows. If a row is full, move Y. If Y is full, move to next plate (X offset).
    
    current_x = plate_margin
    current_y = plate_margin
    row_h = 0.0
    
    for mesh in oriented_meshes:
        bbox = mesh.bounds
        w = bbox[1][0] - bbox[0][0]
        d = bbox[1][1] - bbox[0][1]
        
        # Check if fits in current row
        if current_x + w + plate_margin > plate_width:
            # New row
            current_x = plate_margin
            current_y += row_h + plate_margin
            row_h = 0.0
            
        # Check if fits in current plate (Y)
        if current_y + d + plate_margin > plate_depth:
            # New plate
            plate_offset_x += plate_width + 50.0 # 50mm gap between plates
            current_x = plate_margin
            current_y = plate_margin
            row_h = 0.0
            
        # Place it
        # Align mesh min_x, min_y to current_x, current_y
        # Plus the plate offset
        
        tx = current_x - bbox[0][0] + plate_offset_x
        ty = current_y - bbox[0][1]
        tz = -bbox[0][2] # Ensure on bed
        
        mesh.apply_translation([tx, ty, tz])
        arranged.append(mesh)
        
        current_x += w + plate_margin
        row_h = max(row_h, d)
        
    return arranged

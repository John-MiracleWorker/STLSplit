try:
    from google import genai
    from google.genai import types
except ImportError:
    # Fallback for older installations
    import google.generativeai as genai
    types = None
import matplotlib.pyplot as plt
import numpy as np
import trimesh
import io
import os
from PIL import Image
from typing import List, Optional, Dict, Any

def generate_cut_preview(mesh: trimesh.Trimesh, axis: int, position: float) -> Image.Image:
    """
    Generates a 2D orthogonal projection of the mesh with the cut line.
    """
    # Subsample vertices for speed
    verts = mesh.vertices
    if len(verts) > 50000:
        idx = np.random.choice(len(verts), 50000, replace=False)
        verts = verts[idx]
        
    # Project: We look "down" the axis that is NOT the cut axis?
    # Actually, if we cut along X axis (plane normal X), we want to see YZ plane? 
    # Or better: View perpendicular to the cut.
    # If cut normal is X, we want to look from Z (showing X vs Y) to see where the line falls.
    
    # Let's standardize: Plot the two axes that are most relevant.
    # If cutting along Axis K (0=X,1=Y,2=Z) at Value V:
    # We plot Axis K vs Axis (K+1)%3.
    
    x_idx = axis
    y_idx = (axis + 1) % 3
    
    x_vals = verts[:, x_idx]
    y_vals = verts[:, y_idx]
    
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(x_vals, y_vals, s=0.05, c='black', alpha=0.3)
    
    # Draw cut line
    # The cut is at x_idx = position. Vertical line on this plot.
    ax.axvline(x=position, color='red', linewidth=2, linestyle='--')
    
    ax.axis('off')
    ax.set_title(f"Cut Axis {axis} @ {position:.1f}")
    
    # Save to buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)

def rank_cuts_with_gemini(
    mesh: trimesh.Trimesh, 
    candidates: List[Dict[str, Any]], 
    api_key: str
) -> int:
    """
    Sends candidates to Gemini and returns the index of the best one.
    candidates: List of dicts with 'axis', 'pos', 'score' (math score)
    """
    if not api_key:
        return 0 # Fallback to math best
        
    try:
        # New API: use Client
        client = genai.Client(api_key=api_key)
        
        prompt_parts = [
            "You are an expert 3D printing engineer.",
            "I need to split this object into two parts to fit it in a printer.",
            "I have generated several candidate cut lines (shown in red).",
            "Please analyze these images.",
            "Goal 1: Avoid cutting through critical visual features (like faces, eyes, text) if possible.",
            "Goal 2: Avoid creating thin, fragile walls.",
            "Goal 3: Create two roughly balanced or logical chunks.",
            "Which cut is the BEST? Return ONLY the integer index (0, 1, 2...)."
        ]
        
        contents = ["\n".join(prompt_parts)]
        
        for i, cand in enumerate(candidates):
            img = generate_cut_preview(mesh, cand['axis'], cand['pos'])
            # Convert PIL image to bytes
            buf = io.BytesIO()
            img.save(buf, format='PNG')
            buf.seek(0)
            
            contents.append(f"Image {i} (Index {i}):")
            if types is not None:
                contents.append(types.Part.from_bytes(data=buf.getvalue(), mime_type="image/png"))
            else:
                # Fallback for old API
                contents.append({"mime_type": "image/png", "data": buf.getvalue()})
            
        if len(contents) <= 1:
            return 0

        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=contents
        )
        text = response.text.strip()
        
        # Parse integer
        import re
        match = re.search(r'\d+', text)
        if match:
            best_idx = int(match.group())
            if 0 <= best_idx < len(candidates):
                print(f"Gemini chose index {best_idx}!")
                return best_idx
                
        print(f"Gemini returned unclear response: {text}")
        return 0

    except Exception as e:
        print(f"AI ranking failed: {e}")
        return 0

def generate_section_preview(vertices: np.ndarray) -> Optional[Image.Image]:
    """Generates a 2D plot of the cut cross-section."""
    if len(vertices) == 0:
        return None
        
    try:
        # Project 3D vertices to 2D for plotting
        # Simple projection: subtract mean, SVD for principal components
        mask = np.isfinite(vertices).all(axis=1)
        vertices = vertices[mask]
        if len(vertices) < 3:
            return None
        
        coords = vertices - np.mean(vertices, axis=0)
        
        # Check for degenerate case (all points coincident)
        coord_range = np.ptp(coords, axis=0)
        if np.any(coord_range < 1e-6):
            return None
            
        # Use try-except for SVD as it can fail on degenerate matrices
        try:
            u, s, vh = np.linalg.svd(coords, full_matrices=False)
        except np.linalg.LinAlgError:
            return None
            
        # Check for zero singular values (degenerate projection)
        if np.any(s < 1e-10):
            # Fallback: just use first two coordinate axes
            flat = coords[:, :2]
        else:
            # Project onto first two principal components
            flat = coords @ vh.T[:, :2]
            
        if not np.isfinite(flat).all():
            return None
        
        fig, ax = plt.subplots(figsize=(4, 4))
        # Use fill to show solid area
        from matplotlib.patches import Polygon
        # Note: convex hull might be safer if vertices aren't ordered
        # But for section.vertices from trimesh, they are ordered loops usually
        ax.fill(flat[:, 0], flat[:, 1], c='black', alpha=0.8)
        ax.set_aspect("equal", "box")
        ax.axis('off')
        ax.set_title("Cross Section")
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)
        return Image.open(buf)
    except Exception:
        return None

def recommend_connector_with_gemini(
    section_vertices: np.ndarray,
    api_key: str
) -> str:
    """
    Uses Gemini to look at the section shape and recommend a connector.
    """
    if not api_key:
        return "hex" # Default fallback
        
    if len(section_vertices) < 3:
        return "hex"

    buf = generate_section_preview(section_vertices)
    if buf is None:
        return "hex"

    try:
        # New API usage
        client = genai.Client(api_key=api_key)
        
        prompt_text = (
            "Analyze this 2D cross-section of a 3D printed part cut. "
            "Recommend the best connector type to join these parts. "
            "Options: 'dovetail' (for large, simple geometric sections), "
            "'hex' (for general purpose alignment, default), "
            "'magnet' (if the section is small or needs detachability), "
            "'none' (if too thin/small). "
            "Return ONLY the option name."
        )
        
        # Convert buffer to bytes for the API
        image_bytes = buf.getvalue()
        
        response = client.models.generate_content(
            model="gemini-2.0-flash", # Using the newer model if available, or fallback
            contents=[
                prompt_text,
                types.Part.from_bytes(data=image_bytes, mime_type="image/png")
            ]
        )
        
        text = response.text.strip().lower()
        print(f"Gemini connector recommendation: {text}")
        
        if "dovetail" in text: return "dovetail"
        if "magnet" in text: return "magnet"
        if "none" in text: return "none"
        return "hex"

    except Exception as e:
        print(f"AI connector recommendation failed: {e}")
        return "hex"


def render_multi_view_preview(mesh: trimesh.Trimesh) -> Image.Image:
    """
    Renders 4 views of the mesh: Front, Side, Top, Isometric.
    Returns a composite image for AI analysis.
    """
    verts = mesh.vertices
    if len(verts) > 50000:
        idx = np.random.choice(len(verts), 50000, replace=False)
        verts = verts[idx]
    
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    
    # Front view (X vs Z)
    ax = axes[0, 0]
    ax.scatter(verts[:, 0], verts[:, 2], s=0.05, c='black', alpha=0.3)
    ax.set_title("Front View")
    ax.axis('equal')
    ax.axis('off')
    
    # Side view (Y vs Z)  
    ax = axes[0, 1]
    ax.scatter(verts[:, 1], verts[:, 2], s=0.05, c='black', alpha=0.3)
    ax.set_title("Side View")
    ax.axis('equal')
    ax.axis('off')
    
    # Top view (X vs Y)
    ax = axes[1, 0]
    ax.scatter(verts[:, 0], verts[:, 1], s=0.05, c='black', alpha=0.3)
    ax.set_title("Top View")
    ax.axis('equal')
    ax.axis('off')
    
    # Isometric view (projected)
    ax = axes[1, 1]
    iso_x = verts[:, 0] - verts[:, 1] * 0.5
    iso_y = verts[:, 2] + verts[:, 1] * 0.3
    ax.scatter(iso_x, iso_y, s=0.05, c='black', alpha=0.3)
    ax.set_title("Isometric View")
    ax.axis('equal')
    ax.axis('off')
    
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)


def analyze_model_for_splitting(
    mesh: trimesh.Trimesh,
    build_volume: tuple,  # (x_mm, y_mm, z_mm)
    api_key: str
) -> Dict[str, Any]:
    """
    Uses Gemini Vision to analyze the full 3D model and suggest intelligent split strategies.
    
    Returns a dict with:
    - 'model_type': e.g., 'helmet', 'figurine', 'prop'
    - 'suggested_cuts': list of {'axis': 0/1/2, 'position_pct': 0-1, 'reason': str}
    - 'avoid_areas': list of {'description': str}
    - 'connector_recommendations': {'default': 'hex'/'dovetail'/'magnet'}
    """
    if not api_key:
        return {'model_type': 'unknown', 'suggested_cuts': [], 'avoid_areas': [], 'connector_recommendations': {'default': 'hex'}}
    
    try:
        preview = render_multi_view_preview(mesh)
        buf = io.BytesIO()
        preview.save(buf, format='PNG')
        buf.seek(0)
        
        client = genai.Client(api_key=api_key)
        
        bounds = mesh.bounds
        extents = bounds[1] - bounds[0]
        
        prompt = f"""You are a 3D printing expert. Analyze this 3D model from multiple views.

MODEL INFO:
- Size: {extents[0]:.0f}mm x {extents[1]:.0f}mm x {extents[2]:.0f}mm
- Build volume: {build_volume[0]}mm x {build_volume[1]}mm x {build_volume[2]}mm

TASK: Suggest the best way to split this model to fit the build volume.

Return a JSON object with:
{{
  "model_type": "helmet" or "figurine" or "prop" or "mechanical" or "organic",
  "suggested_cuts": [
    {{"axis": 0, "position_pct": 0.5, "reason": "horizontal cut at neck level"}},
    ...
  ],
  "avoid_cutting": ["face", "visor", "details"],
  "default_connector": "hex" or "dovetail" or "magnet"
}}

AXIS: 0=X (left-right), 1=Y (front-back), 2=Z (up-down)
POSITION_PCT: 0.0 = min, 0.5 = center, 1.0 = max

Return ONLY valid JSON, no markdown."""

        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=[
                prompt,
                types.Part.from_bytes(data=buf.getvalue(), mime_type="image/png")
            ]
        )
        
        text = response.text.strip()
        print(f"AI Model Analysis: {text[:200]}...")
        
        # Parse JSON
        import json
        # Clean up potential markdown
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        
        result = json.loads(text)
        return result
        
    except Exception as e:
        print(f"AI model analysis failed: {e}")
        return {
            'model_type': 'unknown',
            'suggested_cuts': [],
            'avoid_areas': [],
            'connector_recommendations': {'default': 'hex'}
        }

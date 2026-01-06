import streamlit as st
import trimesh
import numpy as np
from pathlib import Path
import tempfile
import os
import shutil
import platform
import matplotlib.pyplot as plt

from stlsplit.config import AppConfig, BuildVolume, ConnectorConfig, SplitConfig, OutputConfig
from stlsplit.pipeline import run_pipeline

def _load_preview_mesh(infile: Path):
    try:
        mesh = trimesh.load_mesh(infile)
    except Exception:
        return None

    if isinstance(mesh, trimesh.Scene):
        if not mesh.geometry:
            return None
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))

    if not isinstance(mesh, trimesh.Trimesh):
        return None

    return mesh

def _render_2d_preview(mesh: trimesh.Trimesh) -> None:
    verts = mesh.vertices
    if len(verts) == 0:
        st.info("Preview unavailable: empty mesh.")
        return

    if len(verts) > 50000:
        idx = np.random.choice(len(verts), 50000, replace=False)
        verts = verts[idx]

    spreads = np.ptp(verts, axis=0)
    axes = np.argsort(spreads)[-2:]
    x_idx, y_idx = int(axes[0]), int(axes[1])

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(verts[:, x_idx], verts[:, y_idx], s=0.05, c="black", alpha=0.3)
    ax.set_aspect("equal", "box")
    ax.axis("off")
    axis_labels = ["X", "Y", "Z"]
    ax.set_title(f"Preview ({axis_labels[x_idx]}-{axis_labels[y_idx]})")
    st.pyplot(fig, clear_figure=True)

def _render_3d_preview(infile: Path) -> None:
    os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")
    import pyvista as pv
    from stpyvista import stpyvista

    pv.OFF_SCREEN = True
    mesh = pv.read(infile)
    plotter = pv.Plotter(window_size=[400, 300], off_screen=True)
    plotter.add_mesh(mesh, color="lightblue", show_edges=True)
    plotter.view_isometric()
    stpyvista(plotter, key="3d_preview")

st.set_page_config(page_title="STLSplit God-Tier", page_icon="✂️", layout="wide")

st.title("✂️ STLSplit: God-Tier 3D Splitter")
st.markdown("""
**The most advanced offline splitter.** Featuring Intelligent Cutting (Math + Gemini AI), 
Support-Aware splitting, Auto-Labeling, and Smart Plating.
""")

# Sidebar
st.sidebar.header("Printer Settings")
bx = st.sidebar.number_input("Build X (mm)", 220, 1000, 220, 10)
by = st.sidebar.number_input("Build Y (mm)", 220, 1000, 220, 10)
bz = st.sidebar.number_input("Build Z (mm)", 100, 1000, 250, 10)

st.sidebar.header("Intelligence")
max_depth = st.sidebar.slider("Max Recursive Splits", 1, 10, 6)
support_w = st.sidebar.slider("Support Weight", 0.0, 2.0, 0.8, help="Favor cuts that create flat bases")
api_key = st.sidebar.text_input("Gemini API Key (Optional)", type="password", help="Enables Semantic Awareness")
use_ai = bool(api_key)

st.sidebar.header("Connectors & Labels")
conn_style = st.sidebar.selectbox("Style", ["auto", "hex", "dovetail", "magnet", "none"], index=0)
conn_count = st.sidebar.slider("Count", 1, 4, 2)
peg_radius = st.sidebar.slider("Peg Radius (mm)", 1.0, 8.0, 4.0, 0.5, help="Smaller = less intrusive")
peg_depth = st.sidebar.slider("Peg Depth (mm)", 2.0, 15.0, 8.0, 0.5, help="How far peg extends into part")
if conn_style != "none":
    st.sidebar.caption("✅ Auto-Labeling Enabled")
    st.sidebar.caption("✅ Keyed Joints Enabled")

# Main Interface
uploaded_file = st.file_uploader("Upload Model (STL/3MF)", type=["stl", "obj", "3mf"])

if uploaded_file:
    # Save temp
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp:
        tmp.write(uploaded_file.getvalue())
        infile = Path(tmp.name)
    
    # Preview
    st.subheader("Preview")
    enable_3d_preview = st.checkbox(
        "Enable 3D preview (experimental)",
        value=False,
        help="Uses VTK via PyVista. On macOS this can crash due to Cocoa window creation.",
    )

    allow_3d_preview = enable_3d_preview
    if enable_3d_preview and platform.system() == "Darwin" and not os.environ.get("STLSPLIT_ENABLE_3D_PREVIEW"):
        st.warning("3D preview is disabled on macOS to avoid VTK Cocoa crashes. Set STLSPLIT_ENABLE_3D_PREVIEW=1 to override.")
        allow_3d_preview = False

    if allow_3d_preview:
        try:
            _render_3d_preview(infile)
        except Exception as e:
            st.warning(f"3D preview unavailable: {e}")
            mesh = _load_preview_mesh(infile)
            if mesh is None:
                st.warning("Preview unavailable (could not read mesh).")
            else:
                _render_2d_preview(mesh)
    else:
        mesh = _load_preview_mesh(infile)
        if mesh is None:
            st.warning("Preview unavailable (could not read mesh).")
        else:
            _render_2d_preview(mesh)

    if st.button("🚀 Process Model", type="primary"):
        with st.spinner("Analyzing geometry & splitting..."):
            try:
                cfg = AppConfig(
                    build_volume=BuildVolume(bx, by, bz),
                    connectors=ConnectorConfig(
                        style=conn_style, 
                        count=conn_count,
                        peg_radius_mm=peg_radius,
                        peg_depth_mm=peg_depth
                    ),
                    split=SplitConfig(
                        max_depth=max_depth, 
                        support_weight=support_w,
                        use_ai=use_ai,
                        ai_key=api_key
                    ),
                    output=OutputConfig(format="3mf", engine="manifold")
                )
                
                out_dir = Path(tempfile.mkdtemp())
                outfile = out_dir / f"{infile.stem}_split.3mf"
                
                res = run_pipeline(
                    infile, outfile, cfg, 
                    repair=True, orient=True, 
                    add_connectors=(conn_style!="none")
                )
                
                st.balloons()
                st.success(f"Split into {res.piece_count} parts!")
                
                # Stats
                c1, c2, c3 = st.columns(3)
                c1.metric("Pieces", res.piece_count)
                c2.metric("Vertices", f"{res.repaired_stats.vertices:,}")
                c3.metric("AI Assisted", "Yes" if use_ai else "No")
                
                with open(outfile, "rb") as f:
                    st.download_button(
                        "📥 Download 3MF (Plated & Labeled)", 
                        f, 
                        file_name=outfile.name,
                        mime="model/3mf"
                    )
                    
            except Exception as e:
                st.error(f"Failed: {e}")

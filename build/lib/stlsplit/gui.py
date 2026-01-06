import streamlit as st
import trimesh
import numpy as np
from pathlib import Path
import tempfile
import os
import shutil

from stlsplit.config import AppConfig, BuildVolume, ConnectorConfig, SplitConfig, OutputConfig
from stlsplit.pipeline import run_pipeline

st.set_page_config(page_title="STLSplit GUI", page_icon="✂️", layout="wide")

st.title("✂️ STLSplit: Intelligent 3D Mesh Splitter")
st.markdown("""
Upload a large 3D model (STL, OBJ, 3MF) and split it into smaller pieces that fit your printer's build volume.
The tool automatically adds connectors and optimizes orientation.
""")

# Sidebar for configuration
st.sidebar.header("Printer Settings")
bx = st.sidebar.number_input("Build X (mm)", value=220, step=10)
by = st.sidebar.number_input("Build Y (mm)", value=220, step=10)
bz = st.sidebar.number_input("Build Z (mm)", value=250, step=10)

st.sidebar.header("Connector Settings")
conn_style = st.sidebar.selectbox("Connector Style", ["auto", "hex", "dovetail", "magnet", "none"], index=0)
conn_count = st.sidebar.slider("Connectors per split", 1, 5, 2)
st.sidebar.markdown("---")
tolerance = st.sidebar.slider("Tolerance (mm)", 0.0, 0.5, 0.15, 0.05)
peg_r = st.sidebar.slider("Peg Radius (mm)", 2.0, 10.0, 4.0)
peg_d = st.sidebar.slider("Peg Depth (mm)", 2.0, 20.0, 8.0)

st.sidebar.header("Intelligence Settings")
max_depth = st.sidebar.slider("Max Recursive Depth", 1, 10, 6)
cut_samples = st.sidebar.slider("Cut Samples per Axis", 1, 20, 5)
support_w = st.sidebar.slider("Support Reduction Weight", 0.0, 2.0, 0.5)

# Main UI
uploaded_file = st.file_uploader("Choose a 3D file", type=["stl", "obj", "3mf"])

if uploaded_file is not None:
    # Save to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_input:
        tmp_input.write(uploaded_file.getvalue())
        tmp_input_path = Path(tmp_input.name)

    st.info(f"Loaded: `{uploaded_file.name}`")
    
    if st.button("🚀 Start Splitting", type="primary"):
        with st.spinner("Processing mesh... (this may take a minute for complex files)"):
            try:
                # Prepare config
                config = AppConfig(
                    build_volume=BuildVolume(x_mm=bx, y_mm=by, z_mm=bz),
                    connectors=ConnectorConfig(
                        style=conn_style,
                        count=conn_count,
                        tolerance_mm=tolerance,
                        peg_radius_mm=peg_r,
                        peg_depth_mm=peg_d
                    ),
                    split=SplitConfig(
                        max_depth=max_depth,
                        cut_samples=cut_samples,
                        support_weight=support_w
                    ),
                    output=OutputConfig(format="3mf", engine="manifold")
                )

                # Output path
                temp_out_dir = Path(tempfile.mkdtemp())
                output_path = temp_out_dir / f"{tmp_input_path.stem}_split.3mf"

                # Run pipeline
                result = run_pipeline(
                    tmp_input_path,
                    output_path,
                    config,
                    repair=True,
                    orient=True,
                    add_connectors=(conn_style != "none")
                )

                st.success(f"Successfully split into {result.piece_count} pieces!")
                
                # Show stats
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Final Piece Count", result.piece_count)
                    st.metric("Total Vertices", f"{result.input_stats.vertices:,}")
                with col2:
                    st.metric("Avg Piece Size", f"{bx}x{by}x{bz}")
                    st.metric("Watertight", "Yes" if result.repaired_stats.watertight else "No")
                with col3:
                    st.metric("Engine", config.output.engine)
                    st.metric("Labels", "Enabled" if conn_style != "none" else "Off")

                st.markdown("### 📦 Output Files")
                # Download button
                for p in result.output_paths:
                    if p.exists():
                        with open(p, "rb") as f:
                            btn = st.download_button(
                                label=f"📂 Download {p.name}",
                                data=f,
                                file_name=p.name,
                                mime="application/octet-stream",
                                key=f"dl_{p.name}"
                            )
                
                # Cleanup
                # os.unlink(tmp_input_path)
                # Note: temp files are usually handled by OS or manually if needed

            except Exception as e:
                st.error(f"Error during splitting: {e}")
                st.exception(e)

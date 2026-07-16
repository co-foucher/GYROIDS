"""
CT scan conversion + a static preview.

STATUS: scaffold. Deliberately does NOT rebuild the existing interactive
slice-scrubbing viewer (gyroid_utils.CT_visualization_window) in the
browser: that viewer's scroll-to-scrub / click-to-inspect interactions
don't translate well to Streamlit's rerun-per-widget-interaction model.
Instead this page handles conversion + a simple static slice preview, and
can launch the existing matplotlib/Qt viewer as a separate process for the
full interactive experience.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import subprocess

import plotly.graph_objects as go
import streamlit as st
import SimpleITK as sitk

from gyroid_utils import CT_scans

from app.state import init_state, get_output_dir

st.set_page_config(page_title="CT Analysis", layout="wide")
init_state()
st.title("CT Scan Analysis")

st.subheader("Convert JPG slices to a .mhd volume")
jpg_pattern = st.text_input("JPG folder or glob pattern (e.g. 'data/*.jpg')", value="")
out_name = st.text_input("Output name (no extension)", value="ct_volume")

if st.button("Convert to MHD") and jpg_pattern:
    with st.spinner("Converting..."):
        out_path = get_output_dir() / out_name
        CT_scans.convert_jpg_to_mhd(jpg_pattern, str(out_path))
    st.success(f"Saved {out_path}.mhd")

st.divider()

st.subheader("Preview a .mhd volume (static mid-slice)")
mhd_path = st.text_input("Path to .mhd file", value="")

if mhd_path:
    if not Path(mhd_path).exists():
        st.error("File not found.")
    else:
        image = sitk.ReadImage(mhd_path)
        array = sitk.GetArrayFromImage(image)  # (z, y, x)

        z_index = st.slider("Z slice", 0, array.shape[0] - 1, array.shape[0] // 2)
        fig = go.Figure(go.Heatmap(z=array[z_index], colorscale="Gray"))
        fig.update_layout(height=500, margin=dict(l=0, r=0, t=20, b=0))
        st.plotly_chart(fig, use_container_width=True)

        st.caption(
            "For interactive scrubbing, curvature coloring, and "
            "click-to-inspect greyvalues, use the full desktop viewer:"
        )
        if st.button("Open interactive CT viewer (separate window)"):
            subprocess.Popen([
                sys.executable, "-c",
                "import SimpleITK as sitk, gyroid_utils.CT_visualization_window as w; "
                f"w.open_window(sitk.ReadImage(r'{mhd_path}'))",
            ])
            st.info(
                "Launching in a separate window - requires a local display "
                "and the matplotlib Qt backend (see CT_visualization_window.py)."
            )

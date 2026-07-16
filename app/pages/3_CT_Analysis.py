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

import os
import subprocess

import numpy as np
import plotly.graph_objects as go
import streamlit as st
import SimpleITK as sitk

from gyroid_utils import CT_scans, mesh_tools

from app.state import init_state, get_output_dir
from app.components.file_picker import browse_file, browse_directory
from app.components.ct_pipeline import render_ct_pipeline
from app.components.mesh_preview import render_mesh_preview

st.set_page_config(page_title="CT Analysis", layout="wide")
init_state()
st.title("CT Scan Analysis")

# ===========================================
# ============ convert to mhd ===============
# ===========================================
col_check, col_title = st.columns([1, 100])
with col_check:
    Convert_to_mhd = st.checkbox(".", value=False)
with col_title:
    st.subheader("Convert slices to a .mhd volume")

if Convert_to_mhd:
    input_format = st.radio("Input format", ["JPG", "DICOM", "TIFF"], horizontal=True)

    st.session_state.setdefault("ct_input_path", "")
    col_path, col_browse = st.columns([5, 1])
    with col_path:
        input_path = st.text_input(
            "Input folder (or glob pattern, e.g. 'data/*.jpg')",
            key="ct_input_path",
        )
    with col_browse:
        st.write("")  # spacer so the button lines up with the text box, not its label
        browse_directory("ct_input_path", title=f"Select {input_format} folder")

    memory_saver = st.checkbox(
        "Downscale to 8-bit (memory saver)", value=True,
        help="Normalizes and converts pixel values to uint8 to save memory. "
             "Uncheck to keep the original bit depth.",
    )

    spacing = None
    if input_format in ("JPG", "TIFF"):
        st.caption("Voxel spacing (mm) - JPG/TIFF files have no embedded spacing metadata.")
        s1, s2, s3 = st.columns(3)
        sx = s1.number_input("Spacing X", value=0.2, min_value=0.0001, format="%.4f")
        sy = s2.number_input("Spacing Y", value=0.2, min_value=0.0001, format="%.4f")
        sz = s3.number_input("Spacing Z", value=0.2, min_value=0.0001, format="%.4f")
        spacing = (sx, sy, sz)
    else:
        st.caption("DICOM voxel spacing is read automatically from the file metadata.")

    out_name = st.text_input("Output name (no extension)", value="ct_volume")

    if st.button("Convert to MHD") and input_path:
        with st.spinner(f"Converting {input_format}..."):
            out_path = get_output_dir() / out_name
            if input_format == "JPG":
                CT_scans.convert_jpg_to_mhd(input_path, str(out_path), spacing=spacing, memory_saver=memory_saver)
            elif input_format == "TIFF":
                CT_scans.convert_tiff_to_mhd(input_path, str(out_path), spacing=spacing, memory_saver=memory_saver)
            else:  # DICOM
                CT_scans.convert_dicomm_to_mhd(input_path, str(out_path), memory_saver=memory_saver)
        st.success(f"Saved {out_path}.mhd")

st.divider()

# ===========================================
# ======== select mhd to work on ============
# ===========================================
st.subheader("Preview a .mhd volume (static mid-slice)")
st.session_state.setdefault("ct_mhd_path", "")
col_path, col_browse = st.columns([5, 1])
with col_path:
    mhd_path = st.text_input("Path to .mhd file", key="ct_mhd_path")
with col_browse:
    st.write("")  # spacer so the button lines up with the text box, not its label
    browse_file(
        "ct_mhd_path",
        title="Select an MHD file",
        filetypes=[("MHD files", "*.mhd"), ("All files", "*.*")],
    )

# ===========================================
# ============== preview mhd ================
# ===========================================
if mhd_path:
    if not Path(mhd_path).exists():
        st.error("File not found.")
    else:
        image = sitk.ReadImage(mhd_path)
        array = sitk.GetArrayFromImage(image)  # (z, y, x)
        mid_z_index = array.shape[0] // 2
        st.write(f"Volume shape: {array.shape} (Z, Y, X)")
        sx, sy, _sz = image.GetSpacing()  # (x, y, z) mm/voxel
        fig = go.Figure(go.Heatmap(z=array[mid_z_index], colorscale="Gray"))
        fig.update_layout(height=500, margin=dict(l=0, r=0, t=20, b=0))
        # Lock the y-axis scale to the x-axis scale, weighted by the
        # true voxel spacing (not just row/column count), so the slice
        # is displayed at its real physical aspect ratio instead of being
        # stretched to fill the plot area - matters most for JPG/TIFF
        # stacks, where in-plane spacing and slice spacing often differ.
        fig.update_yaxes(scaleanchor="x", scaleratio=sy / sx)
        fig.update_xaxes(constrain="domain")
        st.plotly_chart(fig, width="stretch")

        st.caption(
            "For interactive scrubbing, curvature coloring, and "
            "click-to-inspect greyvalues, use the full desktop viewer:"
        )
        if st.button("Open interactive CT viewer (separate window)"):
            # CT_visualization_window.py is written for Jupyter, where the
            # user runs `%matplotlib qt` themselves before calling it - it
            # never sets a backend on its own. Here there's no such magic,
            # so matplotlib falls back to whatever it auto-detects; if
            # Streamlit's own process has MPLBACKEND=Agg set (it often
            # does, being headless-by-default), a plain subprocess would
            # inherit that env var and get the non-interactive Agg
            # backend ("FigureCanvasAgg is non-interactive" warning, no
            # window ever appears). Force QtAgg explicitly, and strip any
            # inherited MPLBACKEND so it can't override that.
            env = os.environ.copy()
            env.pop("MPLBACKEND", None)
            subprocess.Popen([
                    sys.executable, "-c",
                    "import matplotlib; matplotlib.use('QtAgg'); "
                    "import SimpleITK as sitk, gyroid_utils.CT_visualization_window as w; "
                    f"w.open_window(sitk.ReadImage(r'{mhd_path}'))",
                ],
                env=env,
            )
            st.info(
                "Launching in a separate window - requires a local display "
                "and PyQt5/PySide installed (see CT_visualization_window.py)."
            )

        st.divider()

        # ===========================================
        # ============== modify mhd =================
        # ===========================================
        processed = render_ct_pipeline(array, key="ct_pipeline")

        if processed is not None:
            st.subheader("Export processed volume")
            processed_out_name = st.text_input(
                "Processed output name (no extension)",
                value="ct_volume_processed",
                key="ct_pipeline_out_name",
            )
            if st.button("Save processed volume as .mhd", key="ct_pipeline_save_btn"):
                out_image = sitk.GetImageFromArray(processed)
                # Spacing/direction still apply after any of these filters;
                # origin may be slightly off post-crop, but that's a minor
                # detail for a first version - reuses the loaded image's
                # metadata rather than defaulting to identity spacing.
                out_image.SetSpacing(image.GetSpacing())
                out_image.SetOrigin(image.GetOrigin())
                out_image.SetDirection(image.GetDirection())
                out_path = get_output_dir() / processed_out_name
                sitk.WriteImage(out_image, str(out_path.with_suffix(".mhd")))
                st.success(f"Saved {out_path}.mhd")

            st.divider()

            # ===========================================
            # ============= generate mesh ================
            # ===========================================
            st.subheader("Generate 3D mesh")
            # image.GetSpacing() is (x, y, z) mm/voxel; `processed` is
            # (z, y, x) like every array in this page - reverse it so each
            # axis of the marching-cubes coordinate grid gets its OWN
            # spacing below (mesh_spacing[0] for axis 0, etc.), instead of
            # a single scalar. mesh_from_matrix derives its per-axis voxel
            # spacing straight from the deltas between adjacent grid
            # points (see its own spacing computation), so anisotropic
            # voxels (dz != dy != dx, common for CT/optical stacks) come
            # out correctly proportioned rather than being treated as cubic.
            mesh_spacing = image.GetSpacing()[::-1]
            st.caption(
                f"Voxel spacing (dz, dy, dx) = "
                f"({mesh_spacing[0]:.4g}, {mesh_spacing[1]:.4g}, {mesh_spacing[2]:.4g}) mm/voxel - "
                "used directly by marching cubes, so anisotropic voxels aren't stretched or squashed."
            )

            st.session_state.setdefault("ct_mesh", None)
            lo, hi = float(processed.min()), float(processed.max())
            m1, m2 = st.columns(2)
            iso_level = m1.number_input(
                "Iso level", value=(lo + hi) / 2.0, key="ct_mesh_iso_level",
                help="Marching-cubes threshold - the surface is extracted where "
                     "the mask crosses this value (roughly the midpoint for a "
                     "0/255 binary mask).",
            )
            mc_step_size = m2.number_input(
                "Step size", value=1, min_value=1, step=1, key="ct_mesh_step",
                help="Marching-cubes step size - higher is faster but coarser.",
            )
            if st.button("Generate mesh", key="ct_mesh_btn"):
                try:
                    with st.spinner("Extracting isosurface mesh..."):
                        a0, a1, a2 = np.meshgrid(
                            np.arange(processed.shape[0]) * mesh_spacing[0],
                            np.arange(processed.shape[1]) * mesh_spacing[1],
                            np.arange(processed.shape[2]) * mesh_spacing[2],
                            indexing="ij",
                        )
                        verts, faces = mesh_tools.mesh_from_matrix(
                            processed.astype(float), float(iso_level), int(mc_step_size), a0, a1, a2,
                        )
                    st.session_state["ct_mesh"] = (verts, faces)
                    st.success(f"Mesh generated: {len(faces)} faces.")
                except Exception as e:
                    st.session_state["ct_mesh"] = None
                    st.error(f"Mesh generation failed: {e}")

            mesh = st.session_state["ct_mesh"]
            if mesh is not None:
                verts, faces = mesh
                st.markdown("**3D mesh preview**")
                render_mesh_preview(faces, verts, get_output_dir(), key="ct_mesh")
                mesh_out_name = st.text_input(
                    "Mesh output name (no extension)", value="ct_mesh", key="ct_mesh_out_name")
                if st.button("Export mesh as STL", key="ct_mesh_export_btn"):
                    out_path = get_output_dir() / mesh_out_name
                    mesh_tools.export_as_STL(verts, faces, str(out_path) + ".stl")
                    st.success(f"Saved {out_path}.stl")

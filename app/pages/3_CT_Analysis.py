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

import numpy as np
import plotly.graph_objects as go
import streamlit as st
import SimpleITK as sitk

from gyroid_utils import CT_scans, mesh_tools

from app.state import init_state, get_output_dir
from app.components.file_picker import browse_file, browse_directory
from app.components.ct_pipeline import render_ct_pipeline
from app.components.mesh_preview import render_mesh_preview
from app.components.ct_viewer_launcher import render_lightweight_toggle, launch_ct_viewer, CT_histogram

st.set_page_config(page_title="CT Analysis", layout="wide")
init_state()
st.title("CT Scan Analysis")

# ===========================================
# =============== functions =================
# ===========================================

@st.cache_resource(show_spinner="Loading volume...", max_entries=1)
def _load_mhd(path: str):
    """Reads a .mhd volume from disk and returns its array + geometry.

    Memoized by Streamlit and keyed on `path` (a cheap string to hash) -
    this whole page reruns top to bottom on ANY widget interaction
    anywhere on it (e.g. adding a step in the pipeline builder below),
    which used to re-read the full volume from disk every single time
    (see the old `updated_mhd` flag this replaces, which never actually
    worked). Now a rerun with the same path is a cache hit: no disk I/O,
    no SimpleITK decode, just the cached return value.

    Returns spacing/origin/direction as plain tuples rather than the
    sitk.Image itself, since that's all downstream code needs from it.

    st.cache_resource, not st.cache_data: cache_data returns a fresh
    *copy* (via pickle) of the cached array on every single call, cache
    hit or not - and this function gets called on every page rerun (any
    widget interaction anywhere on the page), not just when a new file
    is loaded. On a large volume that meant a full-size copy on every
    click, on top of _run_pipeline's own copy and the SimpleITK
    round-trip inside each step. cache_resource hands back the same
    object by reference instead - no copy, no pickling, dramatically
    less memory and faster.

    That trades away cache_data's automatic "you get your own copy, safe
    to mutate" guarantee: with cache_resource, any in-place mutation of
    this return value corrupts the cached singleton for the rest of the
    session. `array.setflags(write=False)` below converts that from a
    silent-corruption risk into an immediate, loud crash (`ValueError:
    assignment destination is read-only`) the moment anything tries to
    mutate it in place - np.array(array, copy=True), which is exactly
    what _run_pipeline does before any pipeline step touches the data,
    still returns a fully independent, fully writable copy regardless of
    this flag.

    max_entries=1: this cache holds a full CT volume per entry, so
    without a cap, loading a second/third large .mhd file in the same
    session keeps every earlier volume resident too - nothing ever
    evicted them. Capping at 1 means switching files evicts the old
    volume instead of accumulating.
    """
    image = sitk.ReadImage(path)
    array = sitk.GetArrayFromImage(image)  # (z, y, x)
    array.setflags(write=False)
    return array, image.GetSpacing(), image.GetOrigin(), image.GetDirection()

@st.cache_data(show_spinner="Loading midplane figure...")
def _create_midplane_figure(mid_slice: np.ndarray):
    fig = go.Figure(go.Heatmap(z=mid_slice, colorscale="Gray"))
    fig.update_layout(height=500, margin=dict(l=0, r=0, t=20, b=0))
    return fig


@st.cache_data(show_spinner="Loading histogram...")
def _load_histogram(_array: np.ndarray, mhd_path: str, bins: int = 256):
    """Histogram of _array's greyvalues. Keyed on (mhd_path, bins) - not
    on _array itself, since hashing a full CT volume would cost more than
    just recomputing the histogram.

    Caching only helps when `bins` repeats - dragging the "Number of
    bins" slider reruns the page (and this function, on a fresh (path,
    bins) key) on every intermediate value it passes through, not just
    where you release it, so a full recompute over the whole volume can
    happen several times per drag gesture. That's a real cost on a large
    scan, but it's an exact count over every voxel - intentionally not
    subsampled.
    """
    hist, bin_edges = np.histogram(_array.ravel(), bins=bins)
    return hist, bin_edges



# ==================================================
# ============ convert images to mhd ===============
# ==================================================
# ------ toggle title ------
col_check, col_title = st.columns([1, 100])
with col_check:
    Convert_to_mhd = st.checkbox(".", value=False)
with col_title:
    st.subheader("Convert slices to a .mhd volume")

if Convert_to_mhd:
    input_format = st.radio("Input format", ["JPG", "DICOM", "TIFF"], horizontal=True)
    # ------ define path to mhd ------
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

    # ------ define pixel spacing ------
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
    
    # ------ convert ------
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
        # ------ load mhd file ------
        array, spacing, origin, direction = _load_mhd(mhd_path)
        # ------ get mid-plane image ------
        st.write(f"Volume shape: {array.shape} (Z, Y, X)")
        sx, sy, _sz = spacing  # (x, y, z) mm/voxel
        mid_plane = array[array.shape[0] // 2]
        fig = _create_midplane_figure(mid_plane)
        fig.update_layout(height=500, margin=dict(l=0, r=0, t=20, b=0))
        # Lock the y-axis scale to the x-axis scale, weighted by the
        # true voxel spacing (not just row/column count), so the slice
        # is displayed at its real physical aspect ratio instead of being
        # stretched to fill the plot area
        fig.update_yaxes(scaleanchor="x", scaleratio=sy / sx)
        fig.update_xaxes(constrain="domain")
        st.plotly_chart(fig, width="stretch")

        # ------opening full viewer ------
        st.caption(
            "For interactive scrubbing, curvature coloring, and "
            "click-to-inspect greyvalues, use the full desktop viewer:"
        )
        lightweight = render_lightweight_toggle("ct_main_viewer")
        if st.button("Open interactive CT viewer (separate window)"):
            launch_ct_viewer(mhd_path, lightweight)
        
        st.divider()

        # ===========================================
        # ============== histogram  =================
        # ===========================================
        st.subheader("Histogram of greyvalues")
        if make_histogram := st.checkbox("Show greyvalue histogram"):
            bin_amount = st.slider("Number of bins", min_value=10, max_value=1024, value=256, step=10)
            hist, bin_edges = _load_histogram(array, mhd_path=mhd_path, bins=bin_amount)
            CT_histogram(hist, bin_edges)

        st.divider()

        # ===========================================================
        # === processing pipeline (build / export / mesh) - boxed  ===
        # ===========================================================
        # Everything from here down (pipeline builder, export, mesh
        # generation) is one workflow stage built on the loaded volume,
        # so it's boxed in its own bordered container to read as visually
        # distinct from the load/preview/histogram section above - this
        # is the part expected to grow (more step types, more export
        # options), so it gets a clearly bounded home now rather than
        # blending into the rest of the page as it did before.
        with st.container(border=True):
            processed = render_ct_pipeline(array, key="ct_pipeline", spacing=spacing)    #Renders the full pipeline builder (add step / reorder / remove / run) for a loaded CT volume, plus a 2D slice preview of the final result.

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
                    out_image.SetSpacing(spacing)
                    out_image.SetOrigin(origin)
                    out_image.SetDirection(direction)
                    out_path = get_output_dir() / processed_out_name
                    sitk.WriteImage(out_image, str(out_path.with_suffix(".mhd")))
                    st.success(f"Saved {out_path}.mhd")

                st.divider()

                # ===========================================
                # ============= generate mesh ================
                # ===========================================
                st.subheader("Generate 3D mesh")
                # spacing is (x, y, z) mm/voxel; `processed` is (z, y, x) like
                # every array in this page - reverse it so each axis of the
                # marching-cubes coordinate grid gets its OWN spacing below
                # (mesh_spacing[0] for axis 0, etc.), instead of a single
                # scalar. mesh_from_matrix derives its per-axis voxel spacing
                # straight from the deltas between adjacent grid points (see
                # its own spacing computation), so anisotropic voxels
                # (dz != dy != dx, common for CT/optical stacks) come out
                # correctly proportioned rather than being treated as cubic.
                mesh_spacing = spacing[::-1]
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
                            # mesh_from_matrix accepts 1D coordinate arrays
                            # and builds the 3D meshgrid internally only if
                            # it actually needs to (see its x.ndim==1 branch
                            # in mesh_tools.py) - pre-expanding it here with
                            # np.meshgrid was allocating three full-volume
                            # float64 arrays (on top of processed.astype
                            # (float) below) for no benefit. On a large scan
                            # that's the single biggest memory spike in this
                            # page, so pass the cheap 1D arrays instead.
                            a0 = np.arange(processed.shape[0]) * mesh_spacing[0]
                            a1 = np.arange(processed.shape[1]) * mesh_spacing[1]
                            a2 = np.arange(processed.shape[2]) * mesh_spacing[2]
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
                    render_mesh_preview(faces, verts, key="ct_mesh")
                    mesh_out_name = st.text_input(
                        "Mesh output name (no extension)", value="ct_mesh", key="ct_mesh_out_name")
                    if st.button("Export mesh as STL", key="ct_mesh_export_btn"):
                        out_path = get_output_dir() / mesh_out_name
                        mesh_tools.export_as_STL(verts, faces, str(out_path) + ".stl")
                        st.success(f"Saved {out_path}.stl")

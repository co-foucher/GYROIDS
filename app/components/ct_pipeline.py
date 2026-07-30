"""
Ordered, reorderable processing pipeline for a loaded CT volume, built on
top of gyroid_utils.CT_scans's individual filter functions (threshold,
dilate, erode, crop, connected-component, hole/island finding). Kept in
its own component (rather than inline in app/pages/3_CT_Analysis.py) so
that page stays focused on load/convert/preview; this file owns
everything about "let the user apply these functions, in whatever order
they want, with their own parameters".

DESIGN
------
- _STEP_REGISTRY maps a human-readable operation name to its CT_scans
  function, a small parameter spec (used to auto-generate the "add step"
  widgets), and a `call` adapter that normalizes that function's actual
  argument order/naming into a uniform `call(image, params_dict) -> image`
  shape - each CT_scans function has a slightly different signature
  (image first, image last, extra seed coordinates, ...), so the adapter
  is what lets the rest of this module treat every step identically.
- The pipeline itself is just an ordered list of {"label", "params"}
  dicts in st.session_state. Reordering/removing a step doesn't touch any
  image data - "Run pipeline" always replays every step from scratch
  against the original loaded array, so the list can be freely edited
  before (re-)running it.
- watershed_algorithm is deliberately not included here: it needs two
  extra mask inputs (sure_fg, sure_bg) beyond "the current image", which
  doesn't fit this one-image-in-one-image-out step model. Left for a
  future, separate step type that can reference other steps' outputs.
- apply_mask_on_image DOES fit the model despite also needing a second
  image: its mask isn't sourced from another step's output, just a
  separate .mhd file path typed in as a param (see the "Apply mask" step
  below and _load_mask_array), so the uniform call(image, params) -> image
  shape still holds.
- Turning the pipeline result into a 3D mesh lives in
  app/pages/3_CT_Analysis.py, not here - see the "Generate 3D mesh"
  section there, which calls gyroid_utils.mesh_tools.mesh_from_matrix
  directly on this module's returned result.
"""
from typing import Optional

import numpy as np
import plotly.graph_objects as go
import SimpleITK as sitk
import streamlit as st

from gyroid_utils import CT_scans
from app.state import get_output_dir
from app.components.ct_viewer_launcher import render_lightweight_toggle, launch_ct_viewer

__all__ = ["render_ct_pipeline"]


# =====================================================================
# 0 - (reserved)
# 1 - _load_mask_array
# 2 - _STEP_REGISTRY
# 3 - _render_add_step
# 4 - _render_step_list
# 5 - _run_pipeline
# 6 - _build_result_fig
# 7 - _render_result_preview
# 8 - render_ct_pipeline
# =====================================================================

# =====================================================================
# 1) _load_mask_array
# =====================================================================
@st.cache_data(show_spinner="Loading mask...")
def _load_mask_array(path: str) -> np.ndarray:
    """
    Reads a mask volume from disk for the "Apply mask" step.

    Cached by Streamlit and keyed on `path` (a cheap string) - same
    pattern as _load_mhd in app/pages/3_CT_Analysis.py, so re-running the
    pipeline several times with the same mask path (e.g. while tweaking
    an unrelated step's parameters) doesn't re-read the mask file from
    disk on every "Run pipeline" click.
    """
    return sitk.GetArrayFromImage(sitk.ReadImage(path))


# =====================================================================
# 2) _STEP_REGISTRY
# =====================================================================
# Each entry: "help" (shown under the operation picker), "params" (spec
# list used to build the add-step widgets), and "call" (adapter from the
# uniform (image, params_dict) shape to that CT_scans function's actual
# signature). Every adapter copies the array before calling: a couple of
# the underlying functions (apply_threshold) mutate their input in place,
# which would otherwise corrupt the "original array" this module replays
# from on every run.
_STEP_REGISTRY = {
    "Threshold -> binary mask": {
        "help": "segment_from_threshold: pixels within [lower, upper] become 255 (foreground), everything else 0.",
        "params": [
            {"key": "lower_threshold", "label": "Lower threshold", "kind": "float", "default": 0.0},
            {"key": "upper_threshold", "label": "Upper threshold", "kind": "float", "default": 255.0},
        ],
        "call": lambda img, p: CT_scans.segment_from_threshold(
            np.array(img, copy=True), p["lower_threshold"], p["upper_threshold"]),
    },
    "Threshold -> keep range": {
        "help": "apply_threshold: pixels outside [lower, upper] are clipped (relative to lower) - not a binary mask.",
        "params": [
            {"key": "lower_threshold", "label": "Lower threshold", "kind": "float", "default": 0.0},
            {"key": "upper_threshold", "label": "Upper threshold", "kind": "float", "default": 255.0},
        ],
        "call": lambda img, p: CT_scans.apply_threshold(
            np.array(img, copy=True), p["lower_threshold"], p["upper_threshold"]),
    },
    "Dilate": {
        "help": "dilate_filter: grayscale dilation (grows bright regions) with the given kernel radius.",
        "params": [
            {"key": "kernel", "label": "Kernel radius", "kind": "int", "default": 1, "min": 1},
        ],
        "call": lambda img, p: CT_scans.dilate_filter(np.array(img, copy=True), p["kernel"]),
    },
    "Erode": {
        "help": "erode_filter: grayscale erosion (shrinks bright regions) with the given kernel radius.",
        "params": [
            {"key": "kernel", "label": "Kernel radius", "kind": "int", "default": 1, "min": 1},
        ],
        "call": lambda img, p: CT_scans.erode_filter(np.array(img, copy=True), p["kernel"]),
    },
    "Crop": {
        "help": "crop_images: keeps the side of the volume given by 'Keep side', cut at 'Cut coordinate'.",
        "params": [
            {"key": "direction", "label": "Keep side", "kind": "select",
             "options": ["up", "down", "left", "right", "front", "back"], "default": "up"},
            {"key": "point", "label": "Cut coordinate", "kind": "int", "default": 0, "min": 0},
        ],
        "call": lambda img, p: CT_scans.crop_images(p["point"], p["direction"], np.array(img, copy=True)),
    },
    "Connected component (seed point)": {
        "help": "connected_filter: keeps only the region connected to the given (x, y, z) seed point.",
        "params": [
            {"key": "x", "label": "Seed X", "kind": "int", "default": 0, "min": 0},
            {"key": "y", "label": "Seed Y", "kind": "int", "default": 0, "min": 0},
            {"key": "z", "label": "Seed Z", "kind": "int", "default": 0, "min": 0},
        ],
        "call": lambda img, p: CT_scans.connected_filter(p["x"], p["y"], p["z"], np.array(img, copy=True)),
    },
    "Find small holes": {
        "help": "find_small_holes: isolates holes in a binary (foreground=255) image, ranked by size (0=foreground).",
        "params": [
            {"key": "max_hole_size", "label": "Max hole rank", "kind": "int", "default": 1, "min": 0},
        ],
        "call": lambda img, p: CT_scans.find_small_holes(np.array(img, copy=True), p["max_hole_size"]),
    },
    "Find islands": {
        "help": "find_islands: isolates small foreground blobs in a binary (foreground=255) image, ranked by size.",
        "params": [
            {"key": "max_island_size", "label": "Max island rank", "kind": "int", "default": 1, "min": 0},
        ],
        "call": lambda img, p: CT_scans.find_islands(np.array(img, copy=True), p["max_island_size"]),
    },
    "Apply mask": {
        "help": "apply_mask_on_image: pixels outside the mask (mask <= 0) are set to 0. "
                "The mask is loaded from a separate .mhd volume - it must have the same "
                "shape as the volume being processed.",
        "params": [
            {"key": "mask_path", "label": "Mask .mhd path", "kind": "text", "default": ""},
        ],
        "call": lambda img, p: CT_scans.apply_mask_on_image(
            np.array(img, copy=True), _load_mask_array(p["mask_path"])),
    },
}


# =====================================================================
# 3) _render_add_step
# =====================================================================
def _render_add_step(key: str, steps: list) -> None:
    """Renders the operation picker + its parameter widgets + "Add step"."""
    op_label = st.selectbox("Add a step", list(_STEP_REGISTRY.keys()), key=f"{key}_op_select")
    spec = _STEP_REGISTRY[op_label]
    st.caption(spec["help"])

    params = {}
    param_specs = spec["params"]
    cols = st.columns(len(param_specs)) if param_specs else []
    for col, p in zip(cols, param_specs):
        widget_key = f"{key}_param_{op_label}_{p['key']}"
        if p["kind"] == "int":
            params[p["key"]] = col.number_input(
                p["label"], value=p["default"], min_value=p.get("min"), step=1, key=widget_key)
        elif p["kind"] == "float":
            params[p["key"]] = col.number_input(p["label"], value=p["default"], key=widget_key)
        elif p["kind"] == "select":
            params[p["key"]] = col.selectbox(
                p["label"], p["options"], index=p["options"].index(p["default"]), key=widget_key)
        elif p["kind"] == "text":
            params[p["key"]] = col.text_input(p["label"], value=p["default"], key=widget_key)

    if st.button("Add step", key=f"{key}_add_btn"):
        steps.append({"label": op_label, "params": dict(params)})


# =====================================================================
# 4) _render_step_list
# =====================================================================
def _render_step_list(key: str, steps: list) -> None:
    """Renders the current pipeline as reorderable/removable rows."""
    if not steps:
        st.info("No steps yet - add one above.")
        return

    for i, step in enumerate(steps):
        param_str = ", ".join(f"{k}={v}" for k, v in step["params"].items())
        title = f"**{i + 1}. {step['label']}**" + (f" ({param_str})" if param_str else "")
        c_label, c_up, c_down, c_remove = st.columns([6, 1, 1, 1])
        c_label.markdown(title)
        if c_up.button("Up", key=f"{key}_up_{i}", disabled=(i == 0)):
            steps[i - 1], steps[i] = steps[i], steps[i - 1]
            st.rerun()
        if c_down.button("Down", key=f"{key}_down_{i}", disabled=(i == len(steps) - 1)):
            steps[i + 1], steps[i] = steps[i], steps[i + 1]
            st.rerun()
        if c_remove.button("Remove", key=f"{key}_remove_{i}"):
            steps.pop(i)
            st.rerun()


# =====================================================================
# 5) _run_pipeline
# =====================================================================
def _run_pipeline(array: np.ndarray, steps: list) -> np.ndarray:
    """Replays every step, in order, starting from a copy of `array`."""
    current = np.array(array, copy=True)
    for step in steps:
        fn = _STEP_REGISTRY[step["label"]]["call"]
        current = np.asarray(fn(current, step["params"]))
    return current


# =====================================================================
# 6) _build_result_fig
# =====================================================================
@st.cache_data(show_spinner=False)
def _build_result_fig(_mid_slice: np.ndarray, version: int):
    """Builds the mid z-slice Heatmap figure for a pipeline result.

    Cached and keyed ONLY on `version` (an int bumped once per successful
    "Run pipeline" click, see render_ct_pipeline) - the leading
    underscore on `_mid_slice` tells Streamlit to skip hashing that array
    entirely rather than use it as part of the cache key. Without that,
    every rerun (e.g. adding/reordering a step, which doesn't change the
    result) would still pay to hash the slice just to find an unchanged
    cache key; with it, a rerun where `version` hasn't changed is a
    same-key cache hit and this function body doesn't execute at all.
    """
    fig = go.Figure(go.Heatmap(z=_mid_slice, colorscale="Gray"))
    fig.update_layout(height=500, margin=dict(l=0, r=0, t=20, b=0))
    return fig


# =====================================================================
# 7) _render_result_preview
# =====================================================================
def _render_result_preview(array: np.ndarray, result: np.ndarray, version: int, key: str) -> None:
    """Renders the mid z-slice heatmap of the current pipeline result,
    plus the interactive-viewer launcher.

    Called on every rerun where a result exists - unlike the old
    just-ran-gated version, the preview now stays visible while you
    add/reorder/remove steps in between runs. That's safe to do cheaply
    because the figure itself comes from the cached _build_result_fig
    (keyed on `version`, not rebuilt unless the pipeline was actually
    re-run) and `array`/`result` are already in memory - the genuinely
    expensive step (reloading the whole volume from disk) is handled by
    _load_mhd's cache in app/pages/3_CT_Analysis.py, not here.
    """
    st.markdown("**Result preview (mid z-slice)**")
    mid_z_index = array.shape[0] // 2
    fig = _build_result_fig(result[mid_z_index], version)
    st.plotly_chart(fig, width="stretch", key=f"{key}_preview_fig")
    lightweight = render_lightweight_toggle(f"{key}_mask_viewer")
    if st.button("Open interactive CT viewer (mask preview)", key=f"{key}_viewer_btn"):
        temp_path = get_output_dir() / "mask_temp.mhd"
        out_image = sitk.GetImageFromArray(result)
        # Spacing/direction still apply after any of these filters;
        # origin may be slightly off post-crop, but that's a minor
        # detail for a first version - reuses the loaded image's
        # metadata rather than defaulting to identity spacing.
        sitk.WriteImage(out_image, temp_path)
        launch_ct_viewer(str(temp_path), lightweight)


# =====================================================================
# 8) render_ct_pipeline
# =====================================================================
def render_ct_pipeline(array: np.ndarray, key: str = "ct_pipeline") -> Optional[np.ndarray]:
    """
    Renders the full pipeline builder (add step / reorder / remove / run)
    for a loaded CT volume, plus a 2D slice preview of the final result.

    PARAMETERS
    ----------
    array : np.ndarray
        The loaded volume to process, e.g. sitk.GetArrayFromImage(image).
        Never mutated - every run starts from a fresh copy.
    key : str, optional
        Session-state key prefix, so multiple pipelines can coexist on a
        page if ever needed.

    RETURNS
    -------
    result : np.ndarray or None
        The processed volume from the last successful "Run pipeline"
        click, or None if the pipeline hasn't been run yet (e.g. the
        caller can use this to decide whether to show an "Export" button).
    """
    steps_key = f"{key}_steps"
    result_key = f"{key}_result"
    version_key = f"{key}_result_version"
    st.session_state.setdefault(steps_key, [])
    st.session_state.setdefault(result_key, None)
    # Bumped once per successful "Run pipeline" click - the cache key
    # _build_result_fig actually keys on, so the preview figure is only
    # rebuilt when the result genuinely changes, not on every rerun.
    st.session_state.setdefault(version_key, 0)
    steps = st.session_state[steps_key]

    st.subheader("Processing pipeline")

    # Two columns: left is "what's in the pipeline right now" (overview +
    # run), right is "what to add next" followed by the result preview -
    # keeps the add-step form and its widgets from pushing the step list
    # further down the page every time a step is added.
    col_overview, col_add = st.columns([1, 1])

    with col_add:
        st.markdown("**Add a step**")
        _render_add_step(key, steps)
        st.divider()

        # Rendered every rerun (not gated on "just ran") so the preview
        # stays visible while you keep editing the pipeline in between
        # runs - see _render_result_preview and _build_result_fig for how
        # that's kept cheap via caching rather than by hiding the preview.
        result = st.session_state[result_key]
        if result is not None:
            _render_result_preview(array, result, st.session_state[version_key], key)
    
    
    with col_overview:
        st.markdown("**Pipeline overview**")
        _render_step_list(key, steps)
        st.divider()
        if st.button("Run pipeline", key=f"{key}_run_btn", disabled=not steps):
            try:
                with st.spinner(f"Running {len(steps)} step(s)..."):
                    st.session_state[result_key] = _run_pipeline(array, steps)
                st.session_state[version_key] += 1
                st.success(f"Pipeline applied ({len(steps)} step(s)).")
            except Exception as e:
                st.session_state[result_key] = None
                st.error(f"Pipeline failed: {e}")

    return st.session_state[result_key]

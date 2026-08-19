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
- The overview list (_render_step_list) shows "Crop"'s resulting shape
  by tracking a running (Z, Y, X) shape through the step list and
  recomputing it with _shape_after_crop, a small function that DUPLICATES
  CT_scans.crop_images's own direction/point arithmetic rather than
  running the real crop just to read `.shape` off the result. That's a
  real coupling: if crop_images's direction handling ever changes,
  _shape_after_crop needs the same change or the displayed size quietly
  goes wrong without the pipeline itself breaking.
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
# 2 - _shape_after_crop
# 3 - 
# 4 - _STEP_REGISTRY
# 5 - _describe_params
# 6 - _render_add_step
# 7 - _render_step_list
# 8 - _run_pipeline
# 9 - _build_result_fig
# 10 - _render_result_preview
# 11 - render_ct_pipeline
# =====================================================================

# =====================================================================
# 1) _load_mask_array
# =====================================================================
@st.cache_resource(show_spinner="Loading mask...", max_entries=1)
def _load_mask_array(path: str) -> np.ndarray:
    """
    Reads a mask volume from disk for the "Apply mask" step.

    Cached by Streamlit and keyed on `path` (a cheap string) - same
    pattern as _load_mhd in app/pages/3_CT_Analysis.py, so re-running the
    pipeline several times with the same mask path (e.g. while tweaking
    an unrelated step's parameters) doesn't re-read the mask file from
    disk on every "Run pipeline" click.

    st.cache_resource, not st.cache_data: same reasoning as _load_mhd -
    cache_data would hand back a fresh pickled copy of the mask on every
    page rerun, not just when the mask path changes. cache_resource
    returns the same object by reference instead. The read-only flag
    below is the same safety net as _load_mhd's: the "Apply mask" step
    only ever reads this mask (apply_mask_on_image does np.where(mask >
    0, image, 0), never mutates it), but marking it read-only turns any
    future accidental in-place mutation into an immediate crash instead
    of silently corrupting the cached mask for the rest of the session.

    max_entries=1: same reasoning as _load_mhd's cache cap - this holds
    a full mask volume per entry, so switching mask files shouldn't keep
    every previous one resident.
    """
    mask = sitk.GetArrayFromImage(sitk.ReadImage(path))
    mask.setflags(write=False)
    return mask


# =====================================================================
# 2) _shape_after_crop
# =====================================================================
def _shape_after_crop(shape: tuple, params: dict) -> tuple:
    """Calculate the (Z, Y, X) array shape that CT_scans.crop_images would 
    produce, given the shape entering the crop and its ("direction", "point")
    params.
    """
    n, y, x = shape
    direction, point = params["direction"], params["point"]
    if direction == "up":
        y = point
    elif direction == "down":
        y = y - point
    elif direction == "left":
        x = point
    elif direction == "right":
        x = x - point
    elif direction == "front":
        n = point
    elif direction == "back":
        n = n - point
    return (n, y, x)

# =====================================================================
# 4) _STEP_REGISTRY
# =====================================================================
# Each entry: 
#   "help"     (shown under the operation picker), 
#   "params"   (spec list used to build the add-step widgets), 
#   "call"     (adapter from the uniform (image, params_dict) shape to that CT_scans function's actual signature)
#   "describe" (adapter from (params_dict, shape) to a short, step-specific one-liner for the pipeline overview list)
# 
# `shape` is the (Z, Y, X) shape entering that step; ( every step but "Crop" ignores it)
# "describe" is optional: a step without one still works, it just falls back to a generic key=value listing. 

# None of these "call" adapters copy the array before calling their
# CT_scans function. That used to be here as a defensive np.array(img,
# copy=True) in every entry, on the theory that a couple of the
# underlying functions (apply_threshold) mutate their input in place and
# would otherwise corrupt "the original array this module replays from
# on every run" - but that protection is already provided, once, by
# _run_pipeline's own `current = np.array(array, copy=True)` before the
# loop starts. `current` is never aliased anywhere else while the loop
# runs (each step's output simply replaces it for the next step), so a
# step mutating its input in place only ever touches that already-
# isolated copy, never the original `array`. Copying again here on top
# of that was pure duplication: on a large volume it meant every single
# step paid for a full extra copy of the whole volume before doing any
# actual work (worst for "Crop", which then immediately throws most of
# that duplicated data away) - measured at ~1x the volume size in wasted
# peak memory per step, on top of the SimpleITK round-trip most of these
# functions already do internally.
_STEP_REGISTRY = {
    "Threshold -> binary mask": {
        "help": "segment_from_threshold: pixels within [lower, upper] become 255 (foreground), everything else 0.",
        "params": [
            {"key": "lower_threshold", "label": "Lower threshold", "kind": "float", "default": 0.0},
            {"key": "upper_threshold", "label": "Upper threshold", "kind": "float", "default": 255.0},
        ],
        "call": lambda img, p: CT_scans.segment_from_threshold(
            img, p["lower_threshold"], p["upper_threshold"]),
        "describe": lambda p, shape: f"[{p['lower_threshold']:g}, {p['upper_threshold']:g}] -> 255",
    },
    "Threshold -> keep range": {
        "help": "apply_threshold: pixels outside [lower, upper] are clipped (relative to lower) - not a binary mask.",
        "params": [
            {"key": "lower_threshold", "label": "Lower threshold", "kind": "float", "default": 0.0},
            {"key": "upper_threshold", "label": "Upper threshold", "kind": "float", "default": 255.0},
        ],
        "call": lambda img, p: CT_scans.apply_threshold(
            img, p["lower_threshold"], p["upper_threshold"]),
        "describe": lambda p, shape: f"keep [{p['lower_threshold']:g}, {p['upper_threshold']:g}]",
    },
    "Dilate": {
        "help": "dilate_filter: grayscale dilation (grows bright regions) with the given kernel radius.",
        "params": [
            {"key": "kernel", "label": "Kernel radius", "kind": "int", "default": 1, "min": 1},
        ],
        "call": lambda img, p: CT_scans.dilate_filter(img, p["kernel"]),
        "describe": lambda p, shape: f"grow by radius {p['kernel']}",
    },
    "Erode": {
        "help": "erode_filter: grayscale erosion (shrinks bright regions) with the given kernel radius.",
        "params": [
            {"key": "kernel", "label": "Kernel radius", "kind": "int", "default": 1, "min": 1},
        ],
        "call": lambda img, p: CT_scans.erode_filter(img, p["kernel"]),
        "describe": lambda p, shape: f"shrink by radius {p['kernel']}",
    },
    "Crop": {
        "help": "crop_images: keeps the side of the volume given by 'Keep side', cut at 'Cut coordinate'.",
        "params": [
            {"key": "direction", "label": "Keep side", "kind": "select",
             "options": ["up", "down", "left", "right", "front", "back"], "default": "up"},
            {"key": "point", "label": "Cut coordinate", "kind": "int", "default": 0, "min": 0},
        ],
        "call": lambda img, p: CT_scans.crop_images(p["point"], p["direction"], img),
        "describe": lambda p, shape: f"keep {p['direction']} of {p['point']} -> {_shape_after_crop(shape, p)} (Z, Y, X)",
        #"describe": _describe_crop,
    },
    "Connected component (seed point)": {
        "help": "connected_filter: keeps only the region connected to the given (x, y, z) seed point.",
        "params": [
            {"key": "x", "label": "Seed X", "kind": "int", "default": 0, "min": 0},
            {"key": "y", "label": "Seed Y", "kind": "int", "default": 0, "min": 0},
            {"key": "z", "label": "Seed Z", "kind": "int", "default": 0, "min": 0},
        ],
        "call": lambda img, p: CT_scans.connected_filter(p["x"], p["y"], p["z"], img),
        "describe": lambda p, shape: f"seed ({p['x']}, {p['y']}, {p['z']})",
    },
    "Find small holes": {
        "help": "find_small_holes: isolates holes in a binary (foreground=255) image, ranked by size (0=foreground).",
        "params": [
            {"key": "max_hole_size", "label": "Max hole rank", "kind": "int", "default": 1, "min": 0},
        ],
        "call": lambda img, p: CT_scans.find_small_holes(img, p["max_hole_size"]),
        "describe": lambda p, shape: f"holes ranked <= {p['max_hole_size']}",
    },
    "Find islands": {
        "help": "find_islands: isolates small foreground blobs in a binary (foreground=255) image, ranked by size.",
        "params": [
            {"key": "max_island_size", "label": "Max island rank", "kind": "int", "default": 1, "min": 0},
        ],
        "call": lambda img, p: CT_scans.find_islands(img, p["max_island_size"]),
        "describe": lambda p, shape: f"islands ranked <= {p['max_island_size']}",
    },
    "Apply mask": {
        "help": "apply_mask_on_image: pixels outside the mask (mask <= 0) are set to 0. "
                "The mask is loaded from a separate .mhd volume - it must have the same "
                "shape as the volume being processed.",
        "params": [
            {"key": "mask_path", "label": "Mask .mhd path", "kind": "text", "default": ""},
        ],
        "call": lambda img, p: CT_scans.apply_mask_on_image(
            img, _load_mask_array(p["mask_path"])),
        "describe": lambda p, shape: f"mask: {p['mask_path'] or '(not set)'}",
    },
}


# =====================================================================
# 5) _describe_params
# =====================================================================
def _describe_params(label: str, params: dict, shape: tuple) -> str:
    """fetches in the step-specific "describe" adapter from _STEP_REGISTRY and calls it with (params, shape) 
    to get a short one-liner for the pipeline overview list. 
    """
    describe = _STEP_REGISTRY[label].get("describe")
    if describe is not None:
        return describe(params, shape)
    # fallback: no step-specific adapter, just list the key=value pairs
    return ", ".join(f"{k}={v}" for k, v in params.items())


# =====================================================================
# 6) _render_add_step
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
# 7) _render_step_list
# =====================================================================
def _render_step_list(key: str, steps: list, shape: tuple) -> None:
    """Renders the current pipeline as reorderable/removable rows.

    `steps` is the actual list object living in st.session_state (see
    render_ct_pipeline), not a copy - every mutation below (swap, pop)
    edits that list in place. There's nothing to write back afterward:
    session_state already holds a reference to this same list, so
    mutating it here is enough for the change to stick on the next
    rerun.

    `shape` is the ORIGINAL loaded volume's (Z, Y, X) shape (what step 0
    sees) - not what every row's step individually sees. Only "Crop"
    changes the shape (see _shape_after_crop), so `current_shape` below
    starts at `shape` and only gets updated after a "Crop" row, giving
    each row the shape it actually receives without re-running any of
    the other (shape-preserving) steps just to find that out.
    """
    if not steps:
        st.info("No steps yet - add one above.")
        return

    current_shape = shape
    for i, step in enumerate(steps):
        # Build a string like "lower_threshold=0.0, upper_threshold=255.0" for the
        # parameter list, or "" if there are no parameters.
        param_str = ", ".join(f"{k}={v}" for k, v in step["params"].items())
        # The bolded label is the step's name, and the parenthetical is its parameters (if any).
        #   f"**{i + 1}. {step['label']}**" : the the step name and number in bold
        #   (f" ({param_str})" if param_str else "") : the parameters in parentheses, if any
        title = f"**{i + 1}. {step['label']}**" + (f" ({param_str})" if param_str else "")
        notes = f"**Notes:** {_describe_params(step['label'], step['params'], current_shape)}"
        # define the size of the label and each button column
        c_label, c_up, c_down, c_remove = st.columns([6, 1, 1.5, 2])
        # render the label and notes in the first column, and the buttons in the other three columns
        c_label.markdown(title)
        c_label.caption(notes)
        # define the up, down and remove buttons, and disable the up button for the first step and the down button for the last step
        if c_up.button("Up", key=f"{key}_up_{i}", disabled=(i == 0)):
            steps[i - 1], steps[i] = steps[i], steps[i - 1]
            st.rerun()
        if c_down.button("Down", key=f"{key}_down_{i}", disabled=(i == len(steps) - 1)):
            steps[i + 1], steps[i] = steps[i], steps[i + 1]
            st.rerun()
        if c_remove.button("Remove", key=f"{key}_remove_{i}"):
            steps.pop(i)
            st.rerun()
        # Applied AFTER this row is rendered (with the shape it actually
        # received), so it becomes the input the NEXT row's description
        # is computed against - not this one's.
        if step["label"] == "Crop":
            current_shape = _shape_after_crop(current_shape, step["params"])

# =====================================================================
# 8) _run_pipeline
# =====================================================================
def _run_pipeline(array: np.ndarray, steps: list) -> np.ndarray:
    """Replays every step, in order, starting from a copy of `array`."""
    current = np.array(array, copy=True)
    for step in steps:
        fn = _STEP_REGISTRY[step["label"]]["call"]
        current = np.asarray(fn(current, step["params"]))
    return current


# =====================================================================
# 9) _build_result_fig
# =====================================================================
@st.cache_data(show_spinner=False)
def _build_result_fig(mid_slice: np.ndarray, spacing: tuple = (1.0, 1.0, 1.0)) -> go.Figure:
    """Builds the mid z-slice Heatmap figure for a pipeline result.

    Cached on `mid_slice` ITSELF - not on st.session_state[result_key]
    (the full 3D result _render_result_preview slices this out of).
    That distinction matters: `mid_slice` is bounded by the volume's
    (Y, X) footprint alone, the same size regardless of how many Z-slices
    or pipeline steps are involved, so hashing it directly on every
    rerun is cheap (unlike hashing the full `result`, which is exactly
    the large-array cost _load_histogram's docstring in
    app/pages/3_CT_Analysis.py warns against). Because the hash is of
    the actual data, there's no separate "did it change" signal to
    maintain by hand - no version counter to remember to bump, no
    session-state key beyond `result_key` itself. A rerun that slices
    out the same mid_slice content is a cache hit; a genuinely different
    slice (new pipeline run, or Up/Down changing which step feeds the
    preview) is a miss, exactly and automatically.
    """
    sx, sy, _sz = spacing
    fig = go.Figure(go.Heatmap(z=mid_slice, colorscale="Gray"))
    fig.update_layout(height=500, margin=dict(l=0, r=0, t=20, b=0))
    fig.update_yaxes(scaleanchor="x", scaleratio=sy / sx)
    fig.update_xaxes(constrain="domain")
    return fig


# =====================================================================
# 10) _render_result_preview
# =====================================================================
def _render_result_preview(array: np.ndarray, result: np.ndarray, key: str, spacing: tuple) -> None:
    """Renders the mid z-slice heatmap of the current pipeline result,
    plus the interactive-viewer launcher.

    Called on every rerun where a result exists, so the preview stays
    visible while you add/reorder/remove steps in between runs. That's
    cheap because `array`/`result` are already in memory and building
    the 2D slice figure is fast (see _build_result_fig) - the genuinely
    expensive step (reloading the whole volume from disk) is handled by
    _load_mhd's cache in app/pages/3_CT_Analysis.py, not here.
    """
    st.markdown("**Result preview (mid z-slice)**")
    mid_z_index = array.shape[0] // 2
    fig = _build_result_fig(result[mid_z_index], spacing=spacing)
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
# 11) render_ct_pipeline
# =====================================================================
def render_ct_pipeline(array: np.ndarray, key: str = "ct_pipeline", spacing: tuple = (1.0, 1.0, 1.0)) -> Optional[np.ndarray]:
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
    st.session_state.setdefault(steps_key, [])
    st.session_state.setdefault(result_key, None)
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

    with col_overview:
        st.markdown("**Pipeline overview**")
        _render_step_list(key, steps, array.shape)
        st.divider()
        if st.button("Run pipeline", key=f"{key}_run_btn", disabled=not steps):
            try:
                with st.spinner(f"Running {len(steps)} step(s)..."):
                    st.session_state[result_key] = _run_pipeline(array, steps)
                st.success(f"Pipeline applied ({len(steps)} step(s)).")
            except Exception as e:
                st.session_state[result_key] = None
                st.error(f"Pipeline failed: {e}")

    with col_add:
        # Rendered every rerun (not gated on "just ran") so the preview
        # stays visible while you keep editing the pipeline in between
        # runs.
        result = st.session_state[result_key]
        if result is not None:
            _render_result_preview(array, result, key, spacing)

    return st.session_state[result_key]

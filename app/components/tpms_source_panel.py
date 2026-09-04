
from typing import Optional

import numpy as np
import streamlit as st

from gyroid_utils.TPMS_classes.tpms_custom import CustomTPMSModel
from app.components.equation_input import evaluate_custom_inputs, EquationError

# ---- Field modes (label -> TPMSModel.compute_field(mode=...) argument) ----
FIELD_MODES = {
    "Distance": "distance",
    "Signed": "signed",
    "Signed (inverted)": "signed_inverse",
    "Band": "band",
}

FIELD_HELPS = {
    "Distance": "The density field is computed as the distance (defined by the thickness) to an iso-surface (defined by the threshold).",
    "Signed": "The density field is computed by implicit_field > threshold.",
    "Signed (inverted)": "The density field is computed by implicit_field < threshold.",
    "Band": "The density field is computed as density_field = (thickness - np.abs(implicit_field)) > threshold. This is a fast approximation of the distance field",
}

# Reverse of FIELD_MODES ("distance" -> "Distance"), for turning the internal
# mode string back into its display label - e.g. for the "ignored in mode X"
# caption in render_thickness(), which only receives field_mode.
FIELD_MODE_LABELS = {v: k for k, v in FIELD_MODES.items()}

# SKELETAL_MODES: threshold defines the solid region directly, no thickness used
# SHEET_MODES: threshold locates a surface that's then thickened
# Also imported by app/pages/1_Generate_TPMS.py for its generate-dispatch branching.
SKELETAL_MODES = ("signed", "signed_inverse")
SHEET_MODES = ("band", "distance")


"""
#=====================================================================================================================
0 - (reserved)
1 - _adapt_resolution
2 - render_field_mode
3 - load_STL
4 - render_threshold
5 - render_thickness
6 - generate_ui_tpms
7 - pad_to_square
#=====================================================================================================================
"""


# =====================================================================
# 1) _adapt_resolution
# =====================================================================
def _adapt_resolution(field:np.ndarray, params) -> np.ndarray:
    """
    ============================================================================
    1) _ADAPT_RESOLUTION
    Resamples an imported field/geometry array to match the requested grid
    resolution, warning the user in the UI that this happened.
    ============================================================================

    PARAMETERS
    ----------
    field : np.ndarray
        Imported voxel array whose shape doesn't match params.resolution.
    params : TPMSParams
        Generation settings bundle; params.resolution gives the target size
        along each axis.

    RETURNS
    -------
    field : np.ndarray
        The array resampled to (params.resolution,) * 3.
    """
    st.warning(f"Imported field shape {field.shape} does not match the expected resolution ({params.resolution}, {params.resolution}, {params.resolution}). Field will be resampled to the requested resolution.")
    from gyroid_utils import voxel_tools
    field = voxel_tools.interpolate_voxel_grid(field, params.resolution, params.resolution, params.resolution)
    return field


# =====================================================================
# 2) render_field_mode
# =====================================================================
def render_field_mode() -> str:
    """
    ============================================================================
    2) RENDER_FIELD_MODE
    Draws the "Field mode" selectbox and returns the selected mode.
    ============================================================================

    PARAMETERS
    ----------
    None

    RETURNS
    -------
    field_mode : str
        One of "distance", "signed", "signed_inverse", "band".
    """
    mode_label = st.selectbox(
        "Field mode", list(FIELD_MODES.keys()), index=0,
        key="field_mode",
        help=FIELD_HELPS[st.session_state.get("field_mode", "Distance")],
    )
    return FIELD_MODES[mode_label]

# =====================================================================
# 3) load_STL
# =====================================================================
@st.cache_data(show_spinner=False)
def load_STL(stl_path: str) -> tuple[np.ndarray, np.ndarray]:
    """
    ============================================================================
    3) LOAD_STL
    Loads an STL file and returns its vertices and faces as numpy arrays.
    ============================================================================

    PARAMETERS
    ----------
    stl_path : str
        Path to the STL file.

    RETURNS
    -------
    verts : np.ndarray
        Array of shape (N, 3) containing the vertex coordinates.
    faces : np.ndarray
        Array of shape (M, 3) containing the indices of the vertices that
        form each triangular face.
    """
    from stl import mesh

    # Load the STL file
    your_mesh = mesh.Mesh.from_file(stl_path)

    # Get the vertices and faces
    verts = your_mesh.vectors.reshape(-1, 3)
    faces = np.arange(len(verts)).reshape(-1, 3)

    return verts, faces

# =====================================================================
# 4) render_threshold
# =====================================================================
def render_threshold(field_mode: str) -> float:
    """
    ============================================================================
    4) RENDER_THRESHOLD
    Draws the threshold/level widget appropriate for field_mode and returns
    its value.
    ============================================================================

    PARAMETERS
    ----------
    field_mode : str
        The value returned by render_field_mode(). Depending on the mode,
        the threshold widget is either a number_input or not drawn at all
        (band mode ignores threshold entirely).

    RETURNS
    -------
    threshold : float
        The level/threshold value.
    """
    if field_mode == "band":
            return 0.0  # Band ignores level/threshold entirely - no widget
    else:
        return st.number_input(
            "Field threshold",
            value=0.0,
            key="tpms_threshold",
        )


# =====================================================================
# 5) render_thickness
# =====================================================================
def render_thickness(
    field_mode: str,
    draw_widget: bool = True,
    thickness_source_desc: str = "the Thickness input above",
) -> Optional[float]:
    """
    ============================================================================
    5) RENDER_THICKNESS
    Draws the "Thickness" widget,
    when the caller already has its own thickness source - shows a caption
    instead.
    ============================================================================

    PARAMETERS
    ----------
    field_mode : str
        The value returned by render_field_mode().
    draw_widget : bool, optional
        If True (default - the "Built-in type" branch's case), this
        function draws the "Thickness" number_input for SHEET_MODES, and
        returns 0.0 for SKELETAL_MODES without drawing a widget (those
        modes don't use thickness at all). If False (the "Custom equation"
        / "Import from file" branches, which already collected their own
        thickness elsewhere), no widget is drawn here - instead a caption
        is shown when field_mode isn't in SHEET_MODES, and this function
        returns None so the caller keeps using its own thickness value.
    thickness_source_desc : str, optional
        Only used when draw_widget=False, to phrase the caption. Pass e.g.
        "the Thickness formula above" for the equation branch, or "the
        imported thickness file above" for the file-import branch.

    RETURNS
    -------
    thickness : float or None
        The Thickness value when draw_widget=True (0.0 for SKELETAL_MODES,
        the number_input value for SHEET_MODES). None when
        draw_widget=False - a reminder that this function isn't the source
        of truth for thickness in that case; nothing needs to be done with
        the return value.
    """
    if not draw_widget:
        if field_mode not in SHEET_MODES:
            mode_label = FIELD_MODE_LABELS.get(field_mode, field_mode)
            st.caption(
                f"Note: {thickness_source_desc} is ignored in "
                f"'{mode_label}' mode - it doesn't use thickness at all."
            )
        return None

    if field_mode in SHEET_MODES:
        return st.number_input("Thickness", value=1.0, min_value=0.05, key="tpms_thickness")
    return 0.0


# =====================================================================
# 6) generate_ui_tpms
# =====================================================================
def generate_ui_tpms(
    source: str,
    params,
    BUILTIN_TYPES: dict,
    *,
    type_name: str = None,
    px: float = None,
    py: float = None,
    pz: float = None,
    custom_equation: str = None,
    custom_thickness: str = None,
    field: np.ndarray = None,
    thickness_value=None,
    geometry: np.ndarray = None,
    combination_type: str = None,
) -> None:
    """
    ============================================================================
    6) GENERATE_UI_TPMS
    Builds the TPMS model for the current gui inputs,
    computes its density field and mesh, and stores the result in
    st.session_state["current_model"]
    ============================================================================

    PARAMETERS
    ----------
    source : str
        One of "Built-in type", "Custom equation", "Import from file" - the
        selected value of the "Surface" radio button.
    params : TPMSParams
        The generation settings bundle from 1_Generate_TPMS.py (grid size,
        resolution, field mode, threshold, baseplate thickness, mesh
        simplification/smoothing options). For "Built-in type", also
        supplies params.thickness.
    BUILTIN_TYPES : dict
        Maps a built-in type label (e.g. "Gyroid") to its TPMSModel
        subclass. Only used when source == "Built-in type".
    type_name : str, optional
        Selected built-in type label. Required (non-None) when
        source == "Built-in type".
    px, py, pz : float, optional
        Periods along x/y/z. Required (non-None) when
        source == "Built-in type".
    custom_equation : str, optional
        Custom implicit-field equation string. Required (non-None) when
        source == "Custom equation".
    custom_thickness : str, optional
        Custom thickness formula string, from render_equation_input().
        Required (non-None) when source == "Custom equation".
    field : ndarray, optional
        Imported implicit-field matrix. Required when
        source == "Import from file"; resampled with
        voxel_tools.interpolate_voxel_grid() first if its shape doesn't
        match params.resolution.
    thickness_value : float or ndarray, optional
        Imported thickness matrix (or a 1-element array treated as a
        scalar). Required when source == "Import from file".

    RETURNS
    -------
    None

    RAISES
    ------
    ValueError
        If params.field_mode is not one of the known SKELETAL_MODES/
        SHEET_MODES values.

    NOTES
    -----
    - EquationError raised by evaluate_custom_inputs() is caught internally
      and shown as a Streamlit error message rather than propagating.
    """
    x, y, z = np.meshgrid(
        np.linspace(0, params.size_x, params.resolution),
        np.linspace(0, params.size_y, params.resolution),
        np.linspace(0, params.size_z, params.resolution),
        indexing="ij",)

    try:
        with st.spinner("Computing field and generating mesh..."):
            # ----- Built-in type ------
            if source == "Built-in type":
                model = BUILTIN_TYPES[type_name](x, y, z, px, py, pz, params.thickness)

            # ----- Custom equation ------
            elif source == "Custom equation":
                field, thickness_value = evaluate_custom_inputs(custom_equation, custom_thickness, x, y, z)
                model = CustomTPMSModel(x, y, z, thickness_value, field=field)

            # ----- Import from file ------
            elif source == "Import from file":
                if thickness_value.ndim == 1:
                    thickness_value = thickness_value[0]  # take the first value as a scalar thickness
                if field.shape != (params.resolution, params.resolution, params.resolution):
                    field = _adapt_resolution(field, params)
                model = CustomTPMSModel(x, y, z, thickness_value, field=field)

            # ---- compute density_field ------
            if params.field_mode in SKELETAL_MODES:   # "signed", "signed_inverse"
                model.compute_field(mode=params.field_mode, level=params.threshold)
                mesh_iso_level = 0.0
            elif params.field_mode in SHEET_MODES:   # "band", "distance"
                model.compute_field(mode=params.field_mode, level=params.threshold)
                mesh_iso_level = 0.0
            else:
                raise ValueError(f"Unknown field mode: {params.field_mode}")

            # ----- add baseplates ------
            if params.baseplate_thickness > 0:
                model.add_baseplates(thickness=params.baseplate_thickness)

            # ----- combine with imported geometry ------
            if geometry is not None:
                if geometry.shape != (params.resolution, params.resolution, params.resolution):
                    geometry = _adapt_resolution(geometry, params)
                    geometry = geometry > 0.5
                if combination_type == "Union":
                    model.density_field[geometry > 0] = 1  # union: set solid where geometry is solid
                    #model.density_field[geometry <= 0] = 0  # union: set solid where geometry is solid
                elif combination_type == "Intersection":
                    model.density_field[geometry == 0] = -1  # intersection: set non-solid where geometry is non-solid
                elif combination_type == "Substraction":
                    model.density_field[geometry > 0] = -1       # difference:  first, set solid where geometry is solid
                else:
                    st.error(f"Unknown combination type: {combination_type}")

            # ----- generate mesh ------
            st.session_state["current_field_range"] = (float(model.implicit_field.min()), float(model.implicit_field.max()))
            model.generate_mesh(iso_level=mesh_iso_level)
            if params.auto_smooth:
                model.smooth_mesh(smoothing_factor=params.smoothing_factor)
            target_faces = params.max_faces_count if params.max_faces else params.simplification_factor
            model.simplify_mesh(target_faces=target_faces)
            model.fix_mesh()
            is_valid = model.check_mesh_quality()

        st.session_state["current_model"] = model
        st.session_state["current_equation"] = custom_equation

        if not is_valid:
            st.warning(
                "Generated mesh failed validity checks (not watertight / "
                "self-intersecting). Try a coarser grid, a different "
                "thickness, or a different field mode."
            )
        else:
            st.success(f"Mesh generated: {len(model.faces)} faces.")
    except EquationError as e:
        st.error(f"Equation error: {e}")



# =====================================================================
# 7) pad_to_square
# =====================================================================

def pad_to_square(matrix, pad_value=0):
    """
    ============================================================================
    7) PAD_TO_SQUARE
    Pads a matrix with a constant value so every axis matches the largest
    axis, making the array square along each dimension.
    ============================================================================

    PARAMETERS
    ----------
    matrix : np.ndarray
        The array to pad.
    pad_value : scalar, optional
        The constant fill value used for padding (default = 0).

    RETURNS
    -------
    padded : np.ndarray
        The padded array, with each dimension equal to max(matrix.shape).
    """
    target = max(matrix.shape)
    pad_width = [(0, target - dim) for dim in matrix.shape]
    return np.pad(matrix, pad_width=pad_width, mode="constant", constant_values=pad_value)
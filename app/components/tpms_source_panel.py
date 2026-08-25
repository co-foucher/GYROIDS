
from typing import Optional

import streamlit as st

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


def render_field_mode() -> str:
    """
    Draws the "Field mode" selectbox and returns the selected mode.

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


def render_threshold(field_mode: str) -> float:
    """
    Draws the threshold/level widget appropriate for field_mode and returns
    its value.

    PARAMETERS
    ----------
    field_mode : str
        The value returned by render_field_mode(). depending on the mode, the threshold widget 
        is either a number_input or not drawn at all (band mode ignores threshold entirely).

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
        )


def render_thickness(
    field_mode: str,
    draw_widget: bool = True,
    thickness_source_desc: str = "the Thickness input above",
) -> Optional[float]:
    """
    Draws the "Thickness" widget,
    when the caller already has its own thickness source - shows a caption
    instead.

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
        return st.number_input("Thickness", value=1.0, min_value=0.05)
    return 0.0

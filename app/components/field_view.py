"""
2D slice viewer for a TPMS scalar field (model.v) - meant to sit above the
3D mesh preview so the field can be sanity-checked before/alongside the
extracted surface.

Delegates entirely to gyroid_utils.viz.twod_view_of_matrix(show=False),
which now returns the Plotly figure instead of always calling fig.show()
(fig.show() pops open a separate browser tab/window, which isn't
embeddable in a Streamlit page - see the `show` parameter added to that
function). Reusing it wholesale, rather than rebuilding the heatmap here,
keeps a single source of truth for what a field slice view looks like and
gets the built-in Z-slice slider + Play/Pause animation for free (all
client-side in the browser once rendered, no Streamlit rerun needed to
scrub between slices).
"""
from typing import Optional, Tuple

import numpy as np
import streamlit as st
from scipy.ndimage import uniform_filter1d

from gyroid_utils import viz

# Total pixel budget for the WHOLE animation (all shown Z-frames combined),
_PREVIEW_PIXEL_BUDGET = 4_000_000


"""
#=====================================================================================================================
0 - (reserved)
1 - _build_field_figure
2 - render_field_slice
3 - _render_axis
#=====================================================================================================================
"""


# =====================================================================
# 1) _build_field_figure
# =====================================================================
@st.cache_data(show_spinner="Building field slice view...", max_entries=8)
def _build_field_figure(v: np.ndarray, _x: np.ndarray, _y: np.ndarray, _z: np.ndarray,
                         zmin: Optional[float], zmax: Optional[float]):
    """
    ============================================================================
    1) _BUILD_FIELD_FIGURE
    Pure, cacheable half of render_field_slice: downscale (if needed) and
    build the Plotly figure. Deliberately free of any st.* calls - only the
    (figure, downscale_factor) return value is memoized, so there's no
    widget/UI side effect riding along with the cache.
    ============================================================================

    PARAMETERS
    ----------
    v : (Nx, Ny, Nz) ndarray
        Scalar field to display.
    _x, _y, _z : ndarray
        Coordinate grids matching v.shape.
    zmin, zmax : float or None
        Colorscale range, forwarded to viz.twod_view_of_matrix (falls
        back to min/max-of-v if None).

    RETURNS
    -------
    fig : plotly.graph_objects.Figure
        The 2D slice-view figure.
    factor : int
        The downscaling factor actually applied along Z (1 if none).
    """
    # extract the shape of the field matrix (Nx, Ny, Nz)
    nx, ny, nz = v.shape
    # Compute the maximum number of Z-frames we can show without exceeding the pixel budget. 
    max_frames = max(1, _PREVIEW_PIXEL_BUDGET // (nx * ny))

    # If the number of Z-frames exceeds the maximum allowed, downscale the field along the Z-axis.
    if nz > max_frames:
        # calculate the downscaling factor, must be an integer >= 1
        factor = max(1, nz // max_frames)
        # delete slices along the Z-axis to reduce the number of frames
        #v = uniform_filter1d(v.astype(float), size=factor, axis=2)[:, :, ::factor]
        v = v[:, :, ::factor]
        _x = _x[:, :, ::factor]
        _y = _y[:, :, ::factor]
        _z = _z[:, :, ::factor]
    else:
        factor = 1
    #st.warning(f"v.shape: {v.shape}, _x.shape: {_x.shape}, _y.shape: {_y.shape}, _z.shape: {_z.shape}")
    fig = viz.twod_view_of_matrix(v, _x, _y, _z, zmin=zmin, zmax=zmax, show=False)
    return fig, factor


_AXIS_COLORS = {"x": "#e15759", "y": "#59a14f", "z": "#4e79a7"}

_ORIENTATION_PERMUTATIONS = {
    "x,y": (0, 1, 2),  # default: (X, Y, Z) - slice through Z, image plane is X-Y
    "x,z": (0, 2, 1),  # (X, Z, Y) - slice through Y, image plane is X-Z
    "y,z": (1, 2, 0),  # (Y, Z, X) - slice through X, image plane is Y-Z
    "y,x": (1, 0, 2),  # (Y, X, Z) - slice through Z, image plane is Y-X
    "z,x": (2, 0, 1),  # (Z, X, Y) - slice through Y, image plane is Z-X
    "z,y": (2, 1, 0),  # (Z, Y, X) - slice through X, image plane is Z-Y
}


def render_field_slice(v: np.ndarray,
                       x: np.ndarray,
                       y: np.ndarray,
                       z: np.ndarray,
                       key: str,
                       value_range: Tuple[float, float] = None, 
                       ) -> None:
    """
    ============================================================================
    2) RENDER_FIELD_SLICE
    2D slice viewer for a TPMS scalar field (model.v), with a Z-slice
    slider + Play/Pause animation, sitting above the 3D mesh preview so
    the field can be sanity-checked before/alongside the extracted
    surface.
    ============================================================================

    PARAMETERS
    ----------
    v : (Nx, Ny, Nz) ndarray or None
        Scalar field, e.g. model.v after compute_field(). None shows a
        placeholder instead.
    x, y, z : ndarray
        Coordinate grids matching v.shape (model.x/model.y/model.z).
    key : str
        Unique widget-key suffix (avoids collisions across reruns/pages -
        also used to key the orientation selectbox this function renders).
    value_range : (float, float), optional
        (zmin, zmax) for the colorscale. Pass the cached range computed
        once right after compute_field() to avoid re-scanning the full
        (potentially huge) 3D array - twod_view_of_matrix falls back to
        min/max-of-v itself if not given.

    RETURNS
    -------
    None
    """
    if v is None:
        st.info("Compute a field first to see the 2D slice view.")
        return
    # ask user for orientation of the field's axes
    col_1, col_2 = st.columns([2, 6], vertical_alignment="center")
    with col_2:
        orientation = st.selectbox("orientation", (_ORIENTATION_PERMUTATIONS), key=f"{key}_orientation")
    with col_1:
        _render_axis(orientation)
    # rotate v and its coordinate grids to match the selected orientation
    perm = _ORIENTATION_PERMUTATIONS[orientation]
    if perm != (0, 1, 2):
        grids = (x, y, z)
        v = np.transpose(v, perm)
        x, y, z = (np.transpose(grids[perm[i]], perm) for i in range(3))

    # Build the Plotly figure and downscale if needed to fit the pixel budget.
    nz = v.shape[2]
    zmin, zmax = value_range if value_range is not None else (None, None)
    fig, factor = _build_field_figure(v, x, y, z, zmin, zmax)

    if factor > 1:
        st.caption(
            f"Z resolution is {nz} slices - field will be downscaled by {factor}x "
            f"(blurred + subsampled) for display."
        )
        st.warning("Field preview is downscaled for performance - the final mesh will use the full resolution.")

    st.plotly_chart(fig, width="stretch", key=f"{key}_fieldfig")


# =====================================================================
# 3) _render_axis
# =====================================================================
def _render_axis(axis):
    """
    ============================================================================
    3) _RENDER_AXIS
    Small static SVG diagram shown next to the orientation selectbox: draws
    the two physical axes that make up the heatmap's image plane

    Entirely IA written. But I checked, it's not wrong xD
    ============================================================================

    PARAMETERS
    ----------
    axis : str
        One of _ORIENTATION_PERMUTATIONS's keys, e.g. "x,y" - the image
        plane's (horizontal, vertical) axes, comma-separated.

    RETURNS
    -------
    None
    """
    h, v = axis.split(",")
    s = next(a for a in "xyz" if a not in (h, v))
    # Suffix marker ids with the orientation itself so two _render_axis
    # calls in the same page (e.g. two field views side by side) don't
    # share <marker> ids - duplicate SVG ids are only guaranteed to resolve
    # to the first one defined in the DOM, which would silently mis-color
    # the second diagram's arrowheads.
    slug = axis.replace(",", "")

    def _marker(marker_id, color):
        return (f'<marker id="{marker_id}" markerWidth="6" markerHeight="6" '
                f'refX="5" refY="3" orient="auto"><path d="M0,0 L6,3 L0,6 Z" fill="{color}" /></marker>')

    cx, cy = 50, 55
    svg = (
        f'<svg width="100" height="100" viewBox="0 0 100 100" style="display:block;margin:auto;">'
        f'<defs>{_marker(f"arrowhead-{slug}-h", _AXIS_COLORS[h])}{_marker(f"arrowhead-{slug}-v", _AXIS_COLORS[v])}</defs>'
        f'<line x1="{cx}" y1="{cy}" x2="{cx + 28}" y2="{cy}" stroke="{_AXIS_COLORS[h]}" stroke-width="2.5" '
        f'marker-end="url(#arrowhead-{slug}-h)" />'
        f'<text x="{cx + 38}" y="{cy + 4}" fill="{_AXIS_COLORS[h]}" font-size="13" '
        f'font-family="sans-serif" text-anchor="middle">{h.upper()}</text>'
        f'<line x1="{cx}" y1="{cy}" x2="{cx}" y2="{cy - 28}" stroke="{_AXIS_COLORS[v]}" stroke-width="2.5" '
        f'marker-end="url(#arrowhead-{slug}-v)" />'
        f'<text x="{cx}" y="{cy - 38}" fill="{_AXIS_COLORS[v]}" font-size="13" '
        f'font-family="sans-serif" text-anchor="middle">{v.upper()}</text>'
        f'<circle cx="{cx}" cy="{cy}" r="7" fill="none" stroke="{_AXIS_COLORS[s]}" stroke-width="2" />'
        f'<circle cx="{cx}" cy="{cy}" r="2" fill="{_AXIS_COLORS[s]}" />'
        f'<text x="{cx}" y="{cy + 25}" fill="{_AXIS_COLORS[s]}" font-size="11" '
        f'font-family="sans-serif" text-anchor="middle">{s.upper()} (slice)</text>'
        f'</svg>'
    )
    st.markdown(svg, unsafe_allow_html=True)

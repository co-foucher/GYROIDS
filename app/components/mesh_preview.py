from typing import Optional

import streamlit as st

from gyroid_utils import viz

# Display label -> build_mesh_figure() boolean flag name. Only one of the
# four flags is ever True at a time (mirrors build_mesh_figure's own
# "last one wins / normal is the fallback" behavior when it's called
# directly with more than one flag set).
_COLORSCALE_FLAGS = {
    "Normal (surface direction)": "show_normal_colorscale",
    "Flat": "show_flat_colorscale",
    "Random": "show_random_colorscale",
    "Curvature": "show_curvature_colorscale",
}


"""
#=====================================================================================================================
0 - (reserved)
1 - _build_mesh_figure
2 - render_mesh_preview
#=====================================================================================================================
"""


# =====================================================================
# 1) _build_mesh_figure
# =====================================================================
@st.cache_data(show_spinner="Building mesh preview...", max_entries=8)
def _build_mesh_figure(faces, verts, selected_flag: str):
    """
    ============================================================================
    1) _BUILD_MESH_FIGURE
    Pure, cacheable half of render_mesh_preview: builds the Mesh3d figure
    for one specific color mode. Deliberately free of any st.* calls -
    only the returned figure is memoized, no widget/UI side effect rides
    along with the cache (widgets aren't supported inside cache_data-
    decorated functions - see the write-up in project memory / chat
    history for why the original all-in-one version didn't actually work).
    ============================================================================

    PARAMETERS
    ----------
    faces, verts : ndarray
        Mesh data (as produced by TPMSModel.generate_mesh()).
    selected_flag : str
        One of _COLORSCALE_FLAGS's values (e.g. "show_normal_colorscale"),
        naming which build_mesh_figure() boolean flag to set True.

    RETURNS
    -------
    fig : plotly.graph_objects.Figure
        The Mesh3d figure for the selected color mode.

    NOTES
    -----
    max_entries=8 rather than 1 because a handful of distinct (mesh, colorscale) 
    combos shouldn't evict each other on every call.
    """
    flags = {flag: (flag == selected_flag) for flag in _COLORSCALE_FLAGS.values()}
    return viz.build_mesh_figure(faces, verts, **flags)


# =====================================================================
# 2) render_mesh_preview
# =====================================================================
def render_mesh_preview(faces, verts, key: str, height: int = 600) -> None:
    """
    ============================================================================
    2) RENDER_MESH_PREVIEW
    Embeds a mesh preview (Mesh3d figure + coloring selector) inside a
    Streamlit page.
    ============================================================================

    PARAMETERS
    ----------
    faces, verts : ndarray or None
        Mesh data (as produced by TPMSModel.generate_mesh()). If either is
        None, shows a placeholder instead.
    key : str
        Unique suffix for widget keys (avoids collisions between
        pages/reruns).
    height : int, optional
        Plotly figure height in pixels (default 600).

    RETURNS
    -------
    None
    """
    if faces is None or verts is None:
        st.info("Generate a mesh first to see a preview.")
        return

    label = st.selectbox(
        "Mesh coloring",
        list(_COLORSCALE_FLAGS.keys()),
        key=f"{key}_colorscale",
        help=(
            "Passed straight through to viz.build_mesh_figure(). Curvature "
            "coloring does a per-vertex neighborhood search and is "
            "noticeably slower on large/unsimplified meshes."
        ),
    )
    selected_flag = _COLORSCALE_FLAGS[label]

    fig = _build_mesh_figure(faces, verts, selected_flag)
    fig.update_layout(height=height)
    st.plotly_chart(fig, width="stretch", key=f"{key}_meshfig")

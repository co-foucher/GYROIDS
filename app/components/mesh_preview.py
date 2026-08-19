"""
Embeds a mesh preview inside a Streamlit page.

Reuses gyroid_utils.viz.build_mesh_figure (the same Mesh3d figure-building
core that also backs save_mesh_as_html - see that module for the split)
instead of re-implementing the Plotly figure here, so there's a single
source of truth for what a mesh preview looks like. Also exposes
build_mesh_figure's four colorscale modes as a selector, instead of
hardcoding "normal" as the only option.

Renders via st.plotly_chart(fig) directly rather than going through
save_mesh_as_html()'s .html-file export + an embedded iframe: the mesh
figure never touches disk for the live preview, and the browser reuses the
Plotly runtime Streamlit already loaded once for the page instead of
re-downloading/parsing a multi-MB standalone HTML document on every call.
"""
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


@st.cache_data(show_spinner="Building mesh preview...", max_entries=8)
def _build_mesh_figure(faces, verts, selected_flag: str):
    """
    Pure, cacheable half of render_mesh_preview: builds the Mesh3d figure
    for one specific color mode. Deliberately free of any st.* calls -
    only the returned figure is memoized, no widget/UI side effect rides
    along with the cache (widgets aren't supported inside cache_data-
    decorated functions - see the write-up in project memory / chat
    history for why the original all-in-one version didn't actually work).

    Keyed on `selected_flag` (not just faces/verts) so switching the
    "Mesh coloring" dropdown is a genuine cache miss that rebuilds the
    figure, instead of replaying whatever color mode happened to be
    selected the first time this mesh was rendered.

    max_entries=8 rather than 1 because cache_data's cache is process-
    global (shared across every session/user, not per browser tab) - a
    handful of distinct (mesh, colorscale) combos shouldn't evict each
    other on every call.
    """
    flags = {flag: (flag == selected_flag) for flag in _COLORSCALE_FLAGS.values()}
    return viz.build_mesh_figure(faces, verts, **flags)


def render_mesh_preview(faces, verts, key: str, height: int = 600) -> None:
    """
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

"""
Embeds a mesh preview inside a Streamlit page.

Reuses gyroid_utils.viz.save_mesh_as_html (the same HTML export already
used by the TPMS pipeline / notebooks) instead of re-implementing the
Plotly figure here, so there's a single source of truth for what a mesh
preview looks like. Also exposes save_mesh_as_html's four colorscale
modes as a selector, instead of hardcoding "normal" as the only option.
"""
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

from gyroid_utils import viz

# Display label -> save_mesh_as_html() boolean flag name. Only one of the
# four flags is ever True at a time (mirrors save_mesh_as_html's own
# "last one wins / normal is the fallback" behavior when it's called
# directly with more than one flag set).
_COLORSCALE_FLAGS = {
    "Normal (surface direction)": "show_normal_colorscale",
    "Flat": "show_flat_colorscale",
    "Random": "show_random_colorscale",
    "Curvature": "show_curvature_colorscale",
}


def render_mesh_preview(faces, verts, tmp_dir: Path, key: str, height: int = 600) -> None:
    """
    PARAMETERS
    ----------
    faces, verts : ndarray or None
        Mesh data (as produced by TPMSModel.generate_mesh()). If either is
        None, shows a placeholder instead.
    tmp_dir : Path
        Directory to write the intermediate preview .html file to (the
        session's output directory - see app/state.py).
    key : str
        Unique suffix for the preview file name / widget keys (avoids
        collisions between pages/reruns).
    height : int, optional
        Embedded iframe height in pixels (default 600).
    """
    if faces is None or verts is None:
        st.info("Generate a mesh first to see a preview.")
        return

    label = st.selectbox(
        "Mesh coloring",
        list(_COLORSCALE_FLAGS.keys()),
        key=f"{key}_colorscale",
        help=(
            "Passed straight through to viz.save_mesh_as_html(). Curvature "
            "coloring does a per-vertex neighborhood search and is "
            "noticeably slower on large/unsimplified meshes."
        ),
    )
    selected_flag = _COLORSCALE_FLAGS[label]
    flags = {flag: (flag == selected_flag) for flag in _COLORSCALE_FLAGS.values()}

    html_path = tmp_dir / f"_preview_{key}"
    with st.spinner(f"Building {label.lower()} preview..."):
        viz.save_mesh_as_html(faces, verts, str(html_path), **flags)
    html = html_path.with_suffix(".html").read_text(encoding="utf-8")
    components.html(html, height=height, scrolling=True)

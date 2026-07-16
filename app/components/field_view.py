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

# Above this many Z slices, twod_view_of_matrix's animation (one frame per
# slice, all built up front) starts producing a large/slow figure - the
# same cost it would have in a notebook, just more likely to be hit here
# since the GUI's resolution slider goes much higher than typical
# exploratory notebook use.
_SLOW_NZ_WARNING_THRESHOLD = 100


def render_field_slice(v: Optional[np.ndarray],
                       x: Optional[np.ndarray],
                       y: Optional[np.ndarray],
                       z: Optional[np.ndarray],
                       key: str,
                       value_range: Optional[Tuple[float, float]] = None) -> None:
    """
    PARAMETERS
    ----------
    v : (Nx, Ny, Nz) ndarray or None
        Scalar field, e.g. model.v after compute_field(). None shows a
        placeholder instead.
    x, y, z : ndarray
        Coordinate grids matching v.shape (model.x/model.y/model.z).
    key : str
        Unique widget-key suffix (avoids collisions across reruns/pages).
    value_range : (float, float), optional
        (zmin, zmax) for the colorscale. Pass the cached range computed
        once right after compute_field() to avoid re-scanning the full
        (potentially huge) 3D array - twod_view_of_matrix falls back to
        min/max-of-v itself if not given.
    """
    if v is None:
        st.info("Compute a field first to see the 2D slice view.")
        return

    nz = v.shape[2]
    if nz > _SLOW_NZ_WARNING_THRESHOLD:
        factor = max(1, round(nz / _SLOW_NZ_WARNING_THRESHOLD))
        st.caption(
            f"Z resolution is {nz} slices - field will be downscaled by {factor}x "
            f"(blurred + subsampled) for display."
        )
        # Box-blur along Z first (anti-aliasing) then subsample every `factor`-th
        # slice, so the preview isn't just skipping slices but a proper
        # local-average downscale. x/y/z are sliced identically so the
        # coordinate grids stay aligned with v's new shape.
        v = uniform_filter1d(v, size=factor, axis=2)[:, :, ::factor]
        x = x[:, :, ::factor]
        y = y[:, :, ::factor]
        z = z[:, :, ::factor]
        st.warning("Field preview is downscaled for performance - the final mesh will use the full resolution.")

    zmin, zmax = value_range if value_range is not None else (None, None)
    fig = viz.twod_view_of_matrix(v, x, y, z, zmin=zmin, zmax=zmax, show=False)
    st.plotly_chart(fig, use_container_width=True, key=f"{key}_fieldfig")

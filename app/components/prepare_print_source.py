import numpy as np
import streamlit as st
from plotly.colors import sample_colorscale

from gyroid_utils import voxel_tools

# label -> description for detect_overhangs' 0/1/2/3/4 output (see its
# RETURNS docstring). Order matters: index i's color comes from sampling
# "Portland" (the colorscale twod_view_of_matrix/render_field_slice always
# uses) at position i/4, so this list must stay in 0..4 order.
_OVERHANG_LABELS = ["Empty", "Solid", "Overhang", "Bridge", "Support"]


"""
#=====================================================================================================================
0 - (reserved)
1 - detect_overhangs
2 - render_overhang_legend
3 - minimize_overhangs
#=====================================================================================================================
"""


# =====================================================================
# 1) detect_overhangs
# =====================================================================
@st.cache_data(show_spinner="Detecting overhangs...", max_entries=2)
def detect_overhangs(geometry_matrix: np.ndarray,
                     _X: np.ndarray,
                     _Y: np.ndarray,
                     _Z: np.ndarray,
                     angle: float = 45,
                     bridge: float = 30,
                     add_support_voxels: bool = True) -> np.ndarray:
    """
    ============================================================================
    1) DETECT_OVERHANGS
    Detects overhangs in a voxelized geometry and optionally adds support
    voxels, by delegating to gyroid_utils.voxel_tools.detect_overhangs.
    ============================================================================

    PARAMETERS
    ----------
    geometry_matrix : np.ndarray
        The voxelized geometry matrix.
    _X, _Y, _Z : np.ndarray
        The coordinate grids corresponding to the geometry matrix.
    angle : float, optional
        The overhang angle threshold in degrees (default = 45).
    bridge : float, optional
        The maximum bridge length in mm (default = 30).
    add_support_voxels : bool, optional
        Whether to add support voxels for detected overhangs
        (default = True).

    RETURNS
    -------
    overhang_matrix : np.ndarray
        The geometry matrix with each voxel labeled 0 (empty), 1 (solid),
        2 (overhang), 3 (bridge), or 4 (support) - see _OVERHANG_LABELS.
    """
    # Detect overhangs using the provided parameters
    overhang_matrix = voxel_tools.detect_overhangs(geometry_matrix, _X, _Y, _Z, angle = angle, bridge=bridge, add_support_voxels=add_support_voxels)
    return overhang_matrix


# =====================================================================
# 2) render_overhang_legend
# =====================================================================
def render_overhang_legend() -> None:
    """
    ============================================================================
    2) RENDER_OVERHANG_LEGEND
    Renders a small color legend (Empty/Solid/Overhang/Bridge/Support)
    matching the "Portland" colorscale used to display detect_overhangs'
    output.
    ============================================================================

    PARAMETERS
    ----------
    None

    RETURNS
    -------
    None
    """
    colors = sample_colorscale("Portland", [i / (len(_OVERHANG_LABELS) - 1) for i in range(len(_OVERHANG_LABELS))])
    swatches = "".join(
        f'<span style="display:inline-flex;align-items:center;margin-right:16px;">'
        f'<span style="width:14px;height:14px;background:{color};display:inline-block;'
        f'border-radius:3px;margin-right:5px;border:1px solid rgba(128,128,128,0.4);"></span>'
        f'<span style="font-size:13px;">{label}</span></span>'
        for label, color in zip(_OVERHANG_LABELS, colors)
    )
    st.markdown(f'<div style="display:flex;flex-wrap:wrap;align-items:center;">{swatches}</div>',
                unsafe_allow_html=True)


# =====================================================================
# 3) minimize_overhangs
# =====================================================================
def minimize_overhangs(geometry_matrix: np.ndarray,
                       _X: np.ndarray, _Y: np.ndarray, _Z: np.ndarray,
                       complexity: int = 25,
                       angle: float = 65,
                       bridge: float = 30,
                       verts: np.ndarray = None) -> np.ndarray:
    """
    ============================================================================
    3) MINIMIZE_OVERHANGS
    Finds the best print orientation to minimize overhangs, via
    gyroid_utils.voxel_tools.find_optimal_orientation, and rotates the
    mesh vertices to match.
    ============================================================================

    PARAMETERS
    ----------
    geometry_matrix : np.ndarray
        The voxelized geometry matrix.
    _X, _Y, _Z : np.ndarray
        The coordinate grids corresponding to the geometry matrix.
    complexity : int, optional
        Number of candidate orientations to sample (default = 25).
        Forwarded to find_optimal_orientation's `n`.
    angle : float, optional
        Overhang angle threshold in degrees (default = 65). Forwarded to
        find_optimal_orientation's `overhang_angle`.
    bridge : float, optional
        Maximum bridge length in mm (default = 30). Forwarded to
        find_optimal_orientation's `bridge_size`.
    verts : np.ndarray, optional
        Mesh vertices to rotate into the best orientation, via
        mesh_tools.rotate_STL. Required (non-None) for the vertex
        rotation step to succeed.

    RETURNS
    -------
    best_print_matrix : np.ndarray
        The geometry matrix in the best orientation for printing.
    new_x, new_y, new_z : np.ndarray
        The coordinate grids matching best_print_matrix's orientation.
    verts : np.ndarray
        The input vertices, rotated to match the best orientation.
    """
    # Use the voxel_tools function to find the best orientation
    best_print_matrix, new_x, new_y, new_z, rotation_matrix = voxel_tools.find_optimal_orientation(geometry_matrix, 
                                                                                                   _X, _Y, _Z, 
                                                                                                   n=complexity, 
                                                                                                   overhang_angle = angle, 
                                                                                                   bridge_size=bridge, 
                                                                                                   grid_sample_factor = 1,
                                                                                                   generate_supports=True, 
                                                                                                   give_rotation_matrix=True)
    from gyroid_utils import mesh_tools
    verts = mesh_tools.rotate_STL(verts =verts, rotation = rotation_matrix)
    return best_print_matrix, new_x, new_y, new_z, verts


"""
gyroid_utils
============

just some low-level helpers
"""

from .config import DEFAULT_LINEAR_TOL, DEFAULT_ANGULAR_TOL_DEG
from .logger import logger

"""
#=====================================================================================================================
0 - (reserved)
1 - reload_all
2 - help
3 - pad_to_square
#=====================================================================================================================
"""


# =====================================================================
# 1) reload_all
# =====================================================================
def reload_all():
    """
    ============================================================================
    1) RELOAD_ALL
    Reloads every gyroid_utils submodule, useful for interactive development
    (e.g. in a Jupyter notebook) after editing the source without restarting
    the kernel.
    ============================================================================

    PARAMETERS
    ----------
    None

    RETURNS
    -------
    None

    EXAMPLE
    -------
    >>> import gyroid_utils
    >>> gyroid_utils.reload_all()
    """
    import importlib
    import gyroid_utils

    modules = [
        gyroid_utils,
        gyroid_utils.mesh_tools,
        gyroid_utils.viz,
        gyroid_utils.utils,
        gyroid_utils.io_ops,
        gyroid_utils.config,
        gyroid_utils.abaqus_tools,
        gyroid_utils.TET_mesh_tools,
        gyroid_utils.voxel_tools,
        gyroid_utils.CT_scans,
        gyroid_utils.CT_visualization_window,
    ]

    for m in modules:
        importlib.reload(m)
    print("gyroid_utils: all modules reloaded")


# =====================================================================
# 2) help
# =====================================================================
def help():
    """
    ============================================================================
    2) HELP
    Displays help information about the gyroid_utils package.
    ============================================================================

    PARAMETERS
    ----------
    None

    RETURNS
    -------
    None

    EXAMPLE
    -------
    >>> import gyroid_utils
    >>> gyroid_utils.help()
    """
    help_text = """
    Gyroid Utils Package
    ====================

    Available modules:
    - mesh_tools: Functions for mesh operations
    - viz: Visualization tools
    - io_ops: Input/output operations
    - gyroid: Main gyroid generation functions

     Example usage:
           from gyroid_utils.gyroid import GyroidModel
           x, y, z = np.meshgrid(np.linspace(0,1,64),
                      np.linspace(0,1,64),
                      np.linspace(0,1,64), indexing='ij')
           model = GyroidModel(x, y, z, px=1.0, py=1.0, pz=1.0, thickness=0.2)
    """
    print(help_text)


# =====================================================================
# 3) pad_to_square
# =====================================================================
def pad_to_square(matrix, pad_value=0):
    """
    ============================================================================
    3) PAD_TO_SQUARE
    Pads an array with a constant value (zero by default) so that every
    dimension has the same length, without resampling or cropping any
    existing data. Works for 2D matrices (square) as well as higher-
    dimensional arrays (e.g. a 3D voxel grid padded into a cube).
    ============================================================================

    PARAMETERS
    ----------
    matrix : np.ndarray
        Input array of any shape.
    pad_value : scalar, optional
        Constant value used for padding (default 0).

    RETURNS
    -------
    padded : np.ndarray
        Array with all dimensions equal to max(matrix.shape). The original
        data occupies index 0 along every axis (padding is appended at the
        end of each axis, not centered).

    EXAMPLE
    -------
    >>> pad_to_square(np.ones((2, 5))).shape
    (5, 5)
    >>> pad_to_square(np.ones((3, 4, 2))).shape
    (4, 4, 4)
    """
    import numpy as np

    target = max(matrix.shape)
    pad_width = [(0, target - dim) for dim in matrix.shape]
    return np.pad(matrix, pad_width=pad_width, mode="constant", constant_values=pad_value)

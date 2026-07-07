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

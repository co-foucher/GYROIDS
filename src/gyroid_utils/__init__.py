"""
============================================================================
__INIT__
Package entry point for gyroid_utils. Importing this package runs this file,
which exposes the public submodules and the shared logger.
============================================================================
"""

# --- Version metadata ---------------------------------------------------------
from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("gyroid-utils")   # pip distribution name
except PackageNotFoundError:
    __version__ = "unknown"

# Optionally print or log version
print(f"[gyroid_utils] version {__version__} loaded")


# --- Public submodules --------------------------------------------------------
from . import mesh_tools
from . import viz
from . import io_ops
from . import abaqus_tools
from . import TET_mesh_tools
from . import CT_scans
from . import CT_visualization_window
from . import voxel_overhang_tools
#takes each of those submodules and rebinds it as an attribute of gyroid_utils itself (one level up), 
# so that gyroid_utils.tpms_base (no .TPMS_classes. in the middle) also works.
from .TPMS_classes import tpms_base
from .TPMS_classes import tpms_gyroid
from .TPMS_classes import tpms_schwartzp
from .TPMS_classes import tpms_diamond
from .TPMS_classes import tpms_iwp
from .TPMS_classes import tpms_neovius
from .TPMS_classes import tpms_fischerkochs
from .TPMS_classes import tpms_frd
from .TPMS_classes import tpms_lidinoid
from .TPMS_classes import tpms_splitp
from .logger import logger, set_log_level

#__all__ is a list that defines what gets exported when someone does from package import *.
__all__ = [
    "mesh_tools",
    "viz",
    "io_ops",
    "abaqus_tools",
    "TET_mesh_tools",
    "tpms_base",
    "tpms_gyroid",
    "tpms_schwartzp",
    "tpms_diamond",
    "tpms_iwp",
    "tpms_neovius",
    "tpms_fischerkochs",
    "tpms_frd",
    "tpms_lidinoid",
    "tpms_splitp",
    "voxel_overhang_tools",
    "logger",
    "set_log_level",
    "__version__",
]




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
from . import tpms_base
from . import tpms_gyroid
from . import tpms_schwartzp
from . import tpms_diamond
from . import tpms_iwp
from . import tpms_neovius
from . import tpms_fischerkochs
from . import tpms_frd
from . import tpms_lidinoid
from . import tpms_splitp
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
    "logger",
    "set_log_level",
    "__version__",
]




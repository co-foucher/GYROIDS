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
from . import gyroid
from . import abaqus_tools
from . import TET_mesh_tools
from . import SchwartzP
from . import CT_scans  
from . import CT_visualization_window
from .logger import logger, set_log_level

#__all__ is a list that defines what gets exported when someone does from package import *.
__all__ = [
    "mesh_tools",
    "viz",
    "io_ops",
    "gyroid",
    "abaqus_tools",
    "TET_mesh_tools",
    "logger",
    "set_log_level",
    "SchwartzP",
    "__version__",
]




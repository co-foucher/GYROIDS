"""
__init__.py file lets python recognize that this directory is a package
when the package is import, this code is directly run. Thus it can be used to import every other codes
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
from . import occ_tools
from . import io_ops
from . import gyroid
from .logger import logger, set_log_level


__all__ = [
    "mesh_tools",
    "viz",
    "occ_tools",
    "io_ops",
    "gyroid",
    "logger",
    "set_log_level",
    "__version__",
]




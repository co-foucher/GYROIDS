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
from . import io_tools
from .logger import logger, set_log_level


__all__ = [
    "mesh_tools",
    "viz",
    "occ_tools",
    "io_tools",
    "logger",
    "set_log_level",
    "__version__",
]



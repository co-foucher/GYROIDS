"""
============================================================================
__INIT__
Package entry point for gyroid_utils.

Importing this package is cheap: heavy submodules (mesh processing,
visualization, CT/imaging, simulation helpers, TPMS surface classes) are
NOT imported eagerly anymore. Each one is imported lazily, on first
access, via module-level __getattr__ (PEP 562, Python >= 3.7).

Why: several of these submodules pull in slow-to-import native/scientific
libraries at their own top level (vtk, trimesh, pymeshfix, scikit-image,
plotly, SimpleITK, matplotlib...). Before this change, `import gyroid_utils`
paid for ALL of them unconditionally - including the GUI (app/), where
every single Streamlit page imports gyroid_utils just to reach the shared
logger, so e.g. the Home page (which uses none of this) was still paying
the full cost of SimpleITK/vtk/matplotlib on every first load.

This is transparent to existing call sites: `import gyroid_utils` followed
by `gyroid_utils.mesh_tools.foo(...)` (the pattern used throughout the
notebooks and gyroid_utils.utils.reload_all()) keeps working exactly as
written. The only difference is *when* mesh_tools actually gets imported -
on that first `.mesh_tools` access rather than at `import gyroid_utils`
time - and it's cached after that, so later accesses are a plain,
instant attribute lookup.
============================================================================
"""

import importlib

# --- Version metadata ---------------------------------------------------------
from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("gyroid-utils")   # pip distribution name
except PackageNotFoundError:
    __version__ = "unknown"

# Optionally print or log version
print(f"[gyroid_utils] version {__version__} loaded")

# logger.py has zero heavy dependencies (just stdlib `logging`), and a lot
# of code assumes gyroid_utils.logger / gyroid_utils.set_log_level are just
# there - so these stay normal eager imports, not lazy.
from .logger import logger, set_log_level

# --- Public submodules (lazy) --------------------------------------------------
# name -> dotted import target, relative to this package (see __getattr__).
_LAZY_SUBMODULES = {
    "mesh_tools": ".mesh_tools",
    "viz": ".viz",
    "io_ops": ".io_ops",
    "abaqus_tools": ".abaqus_tools",
    "TET_mesh_tools": ".TET_mesh_tools",
    "CT_scans": ".CT_scans",
    "CT_visualization_window": ".CT_visualization_window",
    "voxel_tools": ".voxel_tools",
    "utils": ".utils",
    "config": ".config",
    "TPMS_classes": ".TPMS_classes",
    # Individual TPMS surface modules, re-bound at the top level (as before)
    # so `gyroid_utils.tpms_gyroid` etc. keeps working without having to go
    # through `gyroid_utils.TPMS_classes` explicitly.
    "tpms_base": ".TPMS_classes.tpms_base",
    "tpms_gyroid": ".TPMS_classes.tpms_gyroid",
    "tpms_schwartzp": ".TPMS_classes.tpms_schwartzp",
    "tpms_diamond": ".TPMS_classes.tpms_diamond",
    "tpms_iwp": ".TPMS_classes.tpms_iwp",
    "tpms_neovius": ".TPMS_classes.tpms_neovius",
    "tpms_fischerkochs": ".TPMS_classes.tpms_fischerkochs",
    "tpms_frd": ".TPMS_classes.tpms_frd",
    "tpms_lidinoid": ".TPMS_classes.tpms_lidinoid",
    "tpms_splitp": ".TPMS_classes.tpms_splitp",
    "tpms_custom": ".TPMS_classes.tpms_custom",
}


def __getattr__(name):
    """
    PEP 562 lazy-import hook. Only runs when `name` isn't already a real
    attribute of this module - i.e. the first time it's accessed. Imports
    the target submodule, caches it directly in this module's namespace
    (so the next access is a plain, instant attribute lookup - no
    re-import), and returns it.
    """
    target = _LAZY_SUBMODULES.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = importlib.import_module(target, __name__)
    globals()[name] = module
    return module


def __dir__():
    return sorted(set(globals()) | set(_LAZY_SUBMODULES))


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
    "voxel_tools",
    "logger",
    "set_log_level",
    "__version__",
]

"""
gyroid_utils
============

A utility library for working with triangle meshes, gyroid structures,
and OpenCascade shells.

Modules
-------
mesh_tools   : Mesh-level geometry operations (connected components, simplification)
viz          : Plotly 3D visualization tools
io_ops       : STL/STEP loaders and exporters
occ_tools    : OpenCascade-based modeling and shape validation
utils        : Generic helpers and shared functionality
config       : Global constants and tolerances
"""

from .config import DEFAULT_LINEAR_TOL, DEFAULT_ANGULAR_TOL_DEG
from .logger import logger

from .mesh_tools import (
    keep_largest_connected_component,
    simplify_mesh,
    calculate_triangle_areas,
)

from .viz import (
    save_mesh_as_html,
    plot_histogram,
)

from .io_ops import (
    load_stl,
    write_step,
)

from .occ_tools import (
    is_shell_closed,
    count_faces,
    make_triangle_face,
    is_valid_shape,
    simplify_shell,
    make_shell_from_faces,
)


__all__ = [
    # mesh tools
    "keep_largest_connected_component",
    "simplify_mesh",
    "calculate_triangle_areas",

    # visualization
    "save_mesh_as_html",
    "plot_histogram",

    # IO
    "load_stl",
    "write_step",

    # OCC tools
    "is_shell_closed",
    "count_faces",
    "make_triangle_face",
    "is_valid_shape",
    "simplify_shell",
    "make_shell_from_faces",

    # config
    "DEFAULT_LINEAR_TOL",
    "DEFAULT_ANGULAR_TOL_DEG",
]



def reload_all():
    import importlib
    import gyroid_utils

    modules = [
        gyroid_utils,
        gyroid_utils.mesh_tools,
        gyroid_utils.viz,
        gyroid_utils.utils,
        gyroid_utils.io_ops,
        gyroid_utils.occ_tools,
        gyroid_utils.logger,
        gyroid_utils.config,
    ]

    for m in modules:
        importlib.reload(m)

    print("gyroid_utils: all modules reloaded")
import numpy as np
import open3d as o3d
from .logger import logger

from OCC.Core.STEPControl import STEPControl_Writer, STEPControl_AsIs
from OCC.Core.IFSelect import IFSelect_RetDone

"""
#=====================================================================================================================
0 - (reserved)
1 - load_stl
2 - write_step
#=====================================================================================================================
"""


#=====================================================================
#1) load_stl
#=====================================================================
def load_stl(filepath):
    """
    ============================================================================
    4) LOAD_STL
    Loads an STL file using Open3D and returns (vertices, faces) as NumPy arrays.
    ============================================================================

    PARAMETERS
    ----------
    filepath : str
        Path to the STL file.

    RETURNS
    -------
    vertices : (V, 3) ndarray (float)
        Vertex coordinate array.
    faces : (F, 3) ndarray (int)
        Triangle index array.

    NOTES
    -----
    - Uses Open3D because it handles ASCII + binary STL robustly.
    - Checks that triangle data exists.

    EXAMPLE
    -------
    >>> verts, faces = load_stl("part.stl")
    >>> print(verts.shape, faces.shape)
    """
    logger.info(f"Loading STL file: {filepath}")

    try:
        mesh = o3d.io.read_triangle_mesh(filepath)
    except Exception as e:
        logger.error(f"Failed to read STL file '{filepath}': {e}", exc_info=True)
        raise

    if not mesh.has_triangles():
        logger.error(f"STL load failed: '{filepath}' contains no triangles.")
        raise ValueError("The STL file contains no triangles.")

    vertices = np.asarray(mesh.vertices, dtype=float)
    faces = np.asarray(mesh.triangles, dtype=int)

    logger.info(
        f"Loaded STL successfully: {vertices.shape[0]} vertices, {faces.shape[0]} faces"
    )
    logger.debug(f"STL vertices sample: {vertices[:3]}")
    logger.debug(f"STL faces sample: {faces[:3]}")

    return vertices, faces




#=====================================================================
#2) WRITE_STEP
#=====================================================================
def write_step(shape, filepath: str):
    """
    ============================================================================
    10) WRITE_STEP
    Writes an OpenCascade shape to a STEP file.
    ============================================================================

    PARAMETERS
    ----------
    shape : TopoDS_Shape
    filepath : str

    RAISES
    ------
    RuntimeError
        If STEP export fails.

    EXAMPLE
    -------
    >>> write_step(my_shape, "model.step")
    """
    logger.info(f"Exporting STEP file to: {filepath}")

    if shape is None or shape.IsNull():
        logger.error("Cannot write STEP: input shape is NULL or invalid.")
        raise ValueError("Cannot write STEP: input shape is NULL or invalid.")

    writer = STEPControl_Writer()

    try:
        writer.Transfer(shape, STEPControl_AsIs)
        stat = writer.Write(filepath)
    except Exception as e:
        logger.error(f"STEP export failed while writing '{filepath}': {e}", exc_info=True)
        raise RuntimeError(f"STEP export failed: {e}") from e

    if stat != IFSelect_RetDone:
        logger.error(
            f"STEP export failed with status {stat} for file '{filepath}'"
        )
        raise RuntimeError(f"STEP export failed with status {stat}")

    logger.info(f"STEP export completed successfully: {filepath}")


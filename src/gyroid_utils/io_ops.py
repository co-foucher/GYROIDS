import numpy as np
import open3d as o3d
from .logger import logger


"""
#=====================================================================================================================
0 - (reserved)
1 - load_stl
2 - 
3 - SAVE MATRICES !!!!!
4 - LOAD MATRICES !!!!!
#=====================================================================================================================
"""


#=====================================================================
#1) load_stl
#=====================================================================
def load_stl(filepath):
    """
    ============================================================================
    1) LOAD_STL
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

    # ------------------------------------------------------------------
    # Validate inputs
    # ------------------------------------------------------------------
    try:
        mesh = o3d.io.read_triangle_mesh(filepath)
    except Exception as e:
        logger.error(f"Failed to read STL file '{filepath}': {e}", exc_info=True)
        raise

    if not mesh.has_triangles():
        logger.error(f"STL load failed: '{filepath}' contains no triangles.")
        raise ValueError("The STL file contains no triangles.")

    # ------------------------------------------------------------------
    # rebuild watertight mesh 
    #    -> stl stores triangles only
    #    -> every shared vertices are duplicated
    # ------------------------------------------------------------------
    mesh.remove_duplicated_vertices()
    mesh.remove_duplicated_triangles()
    mesh.remove_degenerate_triangles()
    mesh.remove_unreferenced_vertices()

    # ------------------------------------------------------------------
    # results
    # ------------------------------------------------------------------
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



#=====================================================================
#3) SAVE_GYROID_MATRICES
#=====================================================================
def save_gyroid_matrices(
    outfile: str,
    Xres: np.ndarray,
    Yres: np.ndarray,
    Zres: np.ndarray,
    Xperiod: np.ndarray,
    Yperiod: np.ndarray,
    Zperiod: np.ndarray,
    thickness: np.ndarray,
    gyroid_field: np.ndarray,
    ):
    """
    ============================================================================
    3) SAVE_GYROID_MATRICES
    Saves gyroid-related matrices to a NumPy .npz archive.
    ============================================================================

    All input arrays must have identical shapes. If the shapes differ,
    the function logs an error and aborts without writing a file.

    PARAMETERS
    ----------
    outfile : str
        Path to the output .npz file.
    Xperiod : np.ndarray
        X-period matrix.
    Yperiod : np.ndarray
        Y-period matrix.
    Zperiod : np.ndarray
        Z-period matrix.
    thickness : np.ndarray
        Thickness matrix.
    gyroid_field: np.ndarray
        the actual field of the gyroid

    RETURNS
    -------
    None

    EXAMPLE
    -------
    >>> save_gyroid_matrices(
    ...     "gyroid_data.npz",
    ...     Xperiod, Yperiod, Zperiod, thickness
    ... )
    """

    arrays = [Xres, Yres, Zres, Xperiod, Yperiod, Zperiod, thickness, gyroid_field]
    same_shape = all(arr.shape == arrays[0].shape for arr in arrays)

    if not same_shape:
        logger.error("Cannot save arrays: input matrices have different shapes.")
        return

    np.savez_compressed(
        outfile,
        Xres=Xres,
        Yres=Yres,
        Zres=Zres,
        Xperiod=Xperiod,
        Yperiod=Yperiod,
        Zperiod=Zperiod,
        thickness=thickness,
        gyroid_field = gyroid_field
    )

    logger.info(f"Saved matrices successfully to: {outfile}")


#=====================================================================
#4) LOAD_GYROID_MATRICES
#=====================================================================
def load_gyroid_matrices(infile: str):
    """
    ============================================================================
    4) LOAD_GYROID_MATRICES
    Loads gyroid-related matrices from a NumPy .npz archive.
    ============================================================================

    The function expects the archive to contain the arrays:
    'Xperiod', 'Yperiod', 'Zperiod', and 'thickness'.

    Errors such as missing files, missing arrays, or corrupted files
    are caught and logged. In these cases, the function returns None.

    PARAMETERS
    ----------
    infile : str
        Path to the input .npz file.

    RETURNS
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] or None
        (Xperiod, Yperiod, Zperiod, thickness, gyroid_field) if successful,
        otherwise None.

    EXAMPLE
    -------
    >>> Xp, Yp, Zp, t = load_gyroid_matrices("gyroid_data.npz")
    """

    try:
        with np.load(infile) as loaded_file:
            Xres = loaded_file["Xres"]
            Yres = loaded_file["Yres"]
            Zres = loaded_file["Zres"]
            Xperiod = loaded_file["Xperiod"]
            Yperiod = loaded_file["Yperiod"]
            Zperiod = loaded_file["Zperiod"]
            thickness = loaded_file["thickness"]
            gyroid_field = loaded_file["gyroid_field"]

    except FileNotFoundError:
        logger.error(f"File not found: {infile}")
        return None

    except KeyError as e:
        logger.error(f"Missing expected array in file {infile}: {e}")
        return None

    except Exception as e:
        logger.error(f"Failed to load matrices from {infile}: {e}")
        return None

    logger.info(f"Matrices successfully loaded from: {infile}")

    return Xres, Yres, Zres, Xperiod, Yperiod, Zperiod, thickness, gyroid_field


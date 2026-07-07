import numpy as np
import trimesh
from typing import Union
from .logger import logger


"""
#=====================================================================================================================
0 - (reserved)
1 - load_stl
2 - (reserved) write_step
3 - save_gyroid_matrices
4 - load_gyroid_matrices
#=====================================================================================================================
"""


#=====================================================================
#1) load_stl
#=====================================================================
def load_stl(filepath):
    """
    ============================================================================
    1) LOAD_STL
    Loads an STL file using trimesh and returns (vertices, faces) as NumPy arrays.
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
    - Uses trimesh to handle ASCII + binary STL.
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
        mesh = trimesh.load(filepath, force="mesh")
    except Exception as e:
        logger.error(f"Failed to read STL file '{filepath}': {e}", exc_info=True)
        raise

    if mesh is None or getattr(mesh, "faces", None) is None or len(mesh.faces) == 0:
        logger.error(f"STL load failed: '{filepath}' contains no triangles.")
        raise ValueError("The STL file contains no triangles.")

    # Clean common STL issues (duplicates, degenerate faces)
    try:
        mesh.process(validate=True)
    except Exception as e:
        logger.warning(f"STL cleanup skipped: {e}")

    # ------------------------------------------------------------------
    # results
    # ------------------------------------------------------------------
    vertices = np.asarray(mesh.vertices, dtype=float)
    faces = np.asarray(mesh.faces, dtype=int)

    logger.info(
        f"Loaded STL successfully: {vertices.shape[0]} vertices, {faces.shape[0]} faces"
    )
    logger.debug(f"STL vertices sample: {vertices[:3]}")
    logger.debug(f"STL faces sample: {faces[:3]}")

    return vertices, faces




#=====================================================================
#2) write_step (reserved)
#=====================================================================



#=====================================================================
#3) save_gyroid_matrices
#=====================================================================
def save_gyroid_matrices(
    outfile: str,
    Xres: np.ndarray,
    Yres: np.ndarray,
    Zres: np.ndarray,
    Xperiod: Union[float, np.ndarray],
    Yperiod: Union[float, np.ndarray],
    Zperiod: Union[float, np.ndarray],
    thickness: Union[float, np.ndarray],
    gyroid_field: np.ndarray,
    ):
    """
    ============================================================================
    3) SAVE_GYROID_MATRICES
    Saves gyroid-related matrices to a NumPy .npz archive.
    ============================================================================

    All input arrays must have identical shapes. If the shapes differ,
    the function logs an error and raises ValueError without writing a file.

    PARAMETERS
    ----------
    outfile : str
        Path to the output file. The archive is always a .npz file, so the
        ".npz" extension is appended automatically if not already present
        - callers don't need to include it (and it won't be doubled up if
        they do).
    Xres : np.ndarray
        X-coordinate grid.
    Yres : np.ndarray
        Y-coordinate grid.
    Zres : np.ndarray
        Z-coordinate grid.
    Xperiod : float or np.ndarray
        X-period. Either a scalar (broadcast to the grid shape) or a
        per-voxel array matching Xres/Yres/Zres's shape.
    Yperiod : float or np.ndarray
        Y-period. Same scalar-or-array rule as Xperiod.
    Zperiod : float or np.ndarray
        Z-period. Same scalar-or-array rule as Xperiod.
    thickness : float or np.ndarray
        Thickness. Same scalar-or-array rule as Xperiod.
    gyroid_field : np.ndarray
        The actual field of the gyroid.

    RETURNS
    -------
    None

    RAISES
    ------
    ValueError
        If the input arrays/periods/thickness cannot be broadcast to a
        common shape, or do not all share the same shape.

    EXAMPLE
    -------
    >>> save_gyroid_matrices(
    ...     "gyroid_data",
    ...     Xres, Yres, Zres, Xperiod, Yperiod, Zperiod, thickness, gyroid_field
    ... )
    """

    # The archive is always .npz; append the extension if the caller didn't
    # include it, rather than requiring them to type it (and without
    # doubling it up if they did include it). load_gyroid_matrices() below
    # does the same normalization, so save/load accept the same bare path.
    outfile = str(outfile)
    if not outfile.endswith(".npz"):
        outfile += ".npz"

    # Xperiod/Yperiod/Zperiod/thickness may be given as a single scalar
    # (GyroidModel/_validate_inputs allow this) or as a per-voxel array. Since
    # GyroidModel.save() passes these straight through from self.px/py/pz/
    # thickness, broadcast scalars up to the grid shape here so both the
    # shape check below and the .npz file itself treat them uniformly,
    # instead of crashing on a bare float with no .shape attribute. An array
    # of the wrong shape still fails the same way as before.
    grid_shape = Xres.shape
    try:
        Xperiod = np.broadcast_to(Xperiod, grid_shape)
        Yperiod = np.broadcast_to(Yperiod, grid_shape)
        Zperiod = np.broadcast_to(Zperiod, grid_shape)
        thickness = np.broadcast_to(thickness, grid_shape)
    except ValueError as e:
        logger.error("Cannot save arrays: input matrices have different shapes.")
        raise ValueError("Cannot save arrays: input matrices have different shapes.") from e

    arrays = [Xres, Yres, Zres, Xperiod, Yperiod, Zperiod, thickness, gyroid_field]
    same_shape = all(arr.shape == arrays[0].shape for arr in arrays)

    if not same_shape:
        logger.error("Cannot save arrays: input matrices have different shapes.")
        raise ValueError("Cannot save arrays: input matrices have different shapes.")

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
#4) load_gyroid_matrices
#=====================================================================
def load_gyroid_matrices(infile: str):
    """
    ============================================================================
    4) LOAD_GYROID_MATRICES
    Loads gyroid-related matrices from a NumPy .npz archive.
    ============================================================================

    The function expects the archive to contain the arrays:
    'Xres', 'Yres', 'Zres', 'Xperiod', 'Yperiod', 'Zperiod', 'thickness',
    and 'gyroid_field'.

    Errors such as missing files, missing arrays, or corrupted files are
    logged and then raised (not swallowed) so callers can't silently end up
    with None.

    PARAMETERS
    ----------
    infile : str
        Path to the input file. The archive is always a .npz file, so the
        ".npz" extension is appended automatically if not already present
        - callers can pass the same bare path used with
        save_gyroid_matrices().

    RETURNS
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        (Xres, Yres, Zres, Xperiod, Yperiod, Zperiod, thickness, gyroid_field).

    RAISES
    ------
    FileNotFoundError
        If infile does not exist.
    KeyError
        If the archive is missing one of the expected arrays.
    RuntimeError
        If the file exists but cannot otherwise be loaded (e.g. corrupted).

    EXAMPLE
    -------
    >>> Xres, Yres, Zres, Xp, Yp, Zp, t, field = load_gyroid_matrices("gyroid_data")
    """

    # Same extension normalization as save_gyroid_matrices(), so callers can
    # pass the identical bare path to both functions.
    infile = str(infile)
    if not infile.endswith(".npz"):
        infile += ".npz"

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
        raise

    except KeyError as e:
        logger.error(f"Missing expected array in file {infile}: {e}. Was it saved with save_gyroid_matrices()?")
        raise

    except Exception as e:
        logger.error(f"Failed to load matrices from {infile}: {e}", exc_info=True)
        raise RuntimeError(f"Failed to load gyroid matrices from '{infile}'") from e

    logger.info(f"Matrices successfully loaded from: {infile}")

    return Xres, Yres, Zres, Xperiod, Yperiod, Zperiod, thickness, gyroid_field

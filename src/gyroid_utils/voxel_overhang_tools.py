import numpy as np
from .logger import logger



"""
#=====================================================================================================================
0 - (reserved)
1 - detect_overhangs
#=====================================================================================================================
"""

# =====================================================================
# 1) detect_overhangs
# =====================================================================
def detect_overhangs(voxel_grid: np.ndarray,
                    x: np.ndarray,
                    y: np.ndarray,
                    z: np.ndarray,
                    angle:float = 45.0) -> np.ndarray:
    """
    ============================================================================
    1) DETECT_OVERHANGS
    Detects overhang voxels in a 3D voxel grid by comparing each slice to the
    previous one and flagging transitions whose implied angle exceeds a given
    threshold.
    ============================================================================

    PARAMETERS
    ----------
    voxel_grid : (nx, ny, nz) ndarray
        3D binary voxel grid, where each element is either 0 (empty) or
        1 (solid).
    x : (nx,) ndarray
        1D array of voxel coordinates along x. Used to compute voxel size.
    y : (ny,) ndarray
        1D array of voxel coordinates along y. Used to compute voxel size.
    z : (nz,) ndarray
        1D array of voxel coordinates along z. Used to compute voxel size.
    angle : float, optional
        Overhang angle threshold in degrees (default = 45.0). Suspicious
        voxels with an implied angle greater than this value are flagged as
        overhangs.

    RETURNS
    -------
    overhang_grid : (nx, ny, nz) ndarray
        Array of the same shape as voxel_grid, where each element is:
        0 (not solid), 1 (solid, not an overhang), or 2 (overhang).

    RAISES
    ------
    ValueError
        If voxel_grid is not a 3D array.
    RuntimeError
        If scipy is not installed (required for the distance transform).

    EXAMPLE
    -------
    >>> overhang_grid = detect_overhangs(voxel_grid, x, y, z, angle=45.0)
    """
    if voxel_grid.ndim != 3:
        raise ValueError("voxel_grid must be a 3D numpy array")
    
    overhang_grid = np.copy(voxel_grid)  # initialize the overhang grid with zeros
    spacing_2D = np.array([x[1]-x[0], y[1]-y[0]])  # calculate the average voxel size in x and y dimensions
    slice_thickness = z[1]-z[0]  # calculate the average voxel size in z dimension

    for i in range(1, voxel_grid.shape[2]):
        suspicious_voxel = np.logical_xor(voxel_grid[:,:,i], voxel_grid[:,:,i-1])  # compare current slice with the previous slice to find where voxel are different (i.e., where there is a transition from solid to empty or vice versa)
        if np.any(suspicious_voxel):  # if there are any suspicious voxels in this slice
            logger.debug(f"Suspicious voxels found in slice {i}. Checking for overhangs...")
        else:
            logger.debug(f"No suspicious voxels found in slice {i}. Skipping overhang check for this slice.")
            continue  # skip to the next slice if there are no suspicious voxels
        #suspicious_voxel = voxel_grid[:,:,i]
        #build the distance transform of the current slice to find the distance of each voxel to the nearest solid voxel in the previous slice
        try:
            from scipy.ndimage import distance_transform_edt
        except Exception as e:
            raise RuntimeError("distance mode requires scipy.ndimage.distance_transform_edt") from e
        # distance transform calculate distance of voxel with value 1 to the closest voxel with value 0, so we need to invert the voxel grid
        # voxels with value 0 stay at 0
        distance_matrix = distance_transform_edt(np.logical_not(voxel_grid[:,:,i-1]), sampling=spacing_2D)
        # only keep the distances of the suspicious voxels the rest is set to 0, i.e 0°
        distance_matrix = np.where(suspicious_voxel==1, distance_matrix, 0) 
        # calculate the angle of each suspicious voxel in degrees
        angle_matrix = np.arctan(distance_matrix / slice_thickness) * 180 / np.pi  # calculate the angle of each suspicious voxel in degrees
        #  mark the suspicious voxels that have an angle greater than the specified angle as overhangs
        overhang_grid[:,:,i] = np.where(angle_matrix > angle, 2, voxel_grid[:,:,i]) 
        logger.debug(f"Slice {i}: {np.sum(overhang_grid[:,:,i])} overhang voxels detected.")
    return overhang_grid
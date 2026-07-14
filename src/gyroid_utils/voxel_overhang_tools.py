import numpy as np
from .logger import logger



"""
#=====================================================================================================================
0 - (reserved)
1 - detect_overhangs
2 - _ray_distance
3 - _detect_bridges
4 - _get_rotation_matrix_from_new_z_direction
5 - _REORIENT_VOXEL_GRID
6 - find_optimal_orientation
#=====================================================================================================================
"""

# =====================================================================
# 1) detect_overhangs
# =====================================================================
def detect_overhangs(voxel_grid: np.ndarray,
                    x: np.ndarray,
                    y: np.ndarray,
                    z: np.ndarray,
                    angle:float = 45.0,
                    bridge:float = 10,
                    n_bridge_angles: int = 8,
                    add_support_voxels: bool = False) -> np.ndarray:
    """
    ============================================================================
    1) DETECT_OVERHANGS
    Detects overhang voxels in a 3D voxel grid by comparing each slice to the
    previous one and flagging transitions whose implied angle exceeds a given
    threshold. Overhang voxels that are actually bridged (supported on two
    roughly opposite sides within the allowed bridging distance) are
    reclassified as safe bridges rather than true overhangs.
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
    bridge : float, optional
        Maximum bridgeable span, in the same physical units as x/y/z
        (default = 10). An overhang voxel found to be supported on two
        roughly opposite sides within this total span is reclassified as a
        bridge instead of a true overhang.
    n_bridge_angles : int, optional
        Number of ray directions (spanning 0-180 degrees, since each is
        tested in both the direction and its opposite) used to search for
        bridge support around each candidate voxel (default = 8, i.e. every
        22.5 degrees).

    RETURNS
    -------
    overhang_grid : (nx, ny, nz) ndarray
        Array of the same shape as voxel_grid, where each element is:
        0 (not solid), 1 (solid, not an overhang), 2 (overhang), or
        3 (bridge - unsupported but spanned between two supports within the
        allowed bridging distance, safe to print without support).

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
    slice_thickness = z[1]-z[0]  # calculate the voxel size in z dimension

    #loop through each slice in the z direction, starting from the second slice (index 1)
    for i in range(1, voxel_grid.shape[2]):
        suspicious_voxel = np.logical_xor(voxel_grid[:,:,i], voxel_grid[:,:,i-1])  # compare current slice with the previous slice to find where voxel are different (i.e., where there is a transition from solid to empty or vice versa)
        if not np.any(suspicious_voxel):  # if there are any suspicious voxels in this slice
            logger.debug(f"No suspicious voxels found in slice {i}. Skipping overhang check for this slice.")
            continue  # skip to the next slice if there are no suspicious voxels
        #suspicious_voxel = voxel_grid[:,:,i]
        #build the distance transform of the current slice to find the distance of each voxel to the nearest solid voxel in the previous slice
        try:
            from scipy.ndimage import distance_transform_edt
        except Exception as e:
            raise RuntimeError("distance mode requires scipy.ndimage.distance_transform_edt") from e
        """
        ======= STEP 1 =======
        # find hanging pixels
        =======================
        """
        # distance transform calculate distance of voxel with value 1 to the closest voxel with value 0, so we need to invert the voxel grid
        # voxels with value 0 stay at 0
        angle_distance_matrix = distance_transform_edt(np.logical_not(voxel_grid[:,:,i-1]), sampling=spacing_2D)
        # only keep the distances of the suspicious voxels the rest is set to 0, i.e 0°
        angle_distance_matrix = np.where(suspicious_voxel==1, angle_distance_matrix, 0) 
        # calculate the angle of each suspicious voxel in degrees
        angle_matrix = np.arctan(angle_distance_matrix / slice_thickness) * 180 / np.pi  # calculate the angle of each suspicious voxel in degrees
        #  mark the suspicious voxels that have an angle greater than the specified angle as overhangs
        overhang_grid[:,:,i] = np.where(angle_matrix > angle, 2, voxel_grid[:,:,i]) 
        logger.debug(f"Slice {i}: {np.sum(overhang_grid[:,:,i])} overhang voxels detected.")
        if bridge == 0:
            voxel_grid[:,:,i][overhang_grid[:,:,i] == 2] = 0
            continue  # if bridge distance is 0, we don't need to check for bridges
        """
        ======= STEP 2 =======
        # find hanging pixels that are in a bridge configuration, i.e., if a voxel is
        # supported on two roughly opposite sides within `bridge` distance in the
        # previous slice, it is not a true overhang (it is a printable bridge).
        =======================
        """
        overhang_idx = np.argwhere(overhang_grid[:, :, i] == 2)
        if overhang_idx.size:
            is_bridge = _detect_bridges(
                support_mask=voxel_grid[:, :, i - 1],
                candidates=overhang_idx,
                spacing_2D=spacing_2D,
                max_span=bridge,
                n_angles=n_bridge_angles,
            )
            bridge_idx = overhang_idx[is_bridge]
            if bridge_idx.size:
                overhang_grid[bridge_idx[:, 0], bridge_idx[:, 1], i] = 3
                logger.debug(f"Slice {i}: {len(bridge_idx)} voxel(s) reclassified as bridges.")
        """
        ======= STEP 3 ========
        # detect voxels where supports could be added and add support voxels 
        =======================
        IT DOES NOT WORK YET
        """
        if add_support_voxels:
            if overhang_idx.size:
                # find the overhang voxels that are not bridges (i.e., true overhangs)
                true_overhang_idx = overhang_idx[~is_bridge]
                if true_overhang_idx.size:
                    # find true-overhang voxels that have no solid voxel in any slice underneath them
                    # (slices 0..i-1, i.e. everything already processed below the current slice i)
                    temp = voxel_grid[true_overhang_idx[:, 0], true_overhang_idx[:, 1], :i]
                    needs_support = np.sum(temp, axis=1) == 0
                    if np.any(needs_support):
                        support_voxels = true_overhang_idx[needs_support]
                        # add support voxels all the way down to the initial slice
                        for j in range(i - 1, -1, -1):
                            voxel_grid[support_voxels[:, 0], support_voxels[:, 1], j] = 1
                        # mark the support voxels in the overhang grid as well, so they are not considered overhangs in future slices
                        overhang_grid[support_voxels[:, 0], support_voxels[:, 1], :1+i] = 1
                        logger.debug(f"Slice {i}: {len(support_voxels)} support voxel(s) added in previous slices.")

        # remove the (true) overhang voxels from the original voxel grid to prevent
        # them from being used as support in the next slice. Bridge voxels (3) are
        # left in place since they are safe to build on.
        voxel_grid[:,:,i][overhang_grid[:,:,i] == 2] = 0
    return overhang_grid


# =====================================================================
# 2) _ray_distance
# =====================================================================
def _ray_distance(mask: np.ndarray,
                   candidates: np.ndarray,
                   theta: float,
                   steps: np.ndarray,
                   spacing_2D: np.ndarray) -> np.ndarray:
    """
    ============================================================================
    2) _RAY_DISTANCE
    Marches outward from every candidate voxel along a single fixed direction
    (in the slice plane) and returns, for each one, the physical distance to
    the first solid voxel hit in `mask` (or np.inf if none is found within
    `steps`). 
    It can take several candidate in order to vectorize the function...
    ============================================================================

    PARAMETERS
    ----------
    mask : (nx, ny) ndarray
        Binary mask (previous slice) to search for support in.
    candidates : (n, 2) ndarray
        Row/col (x/y) indices of the query voxels.
    theta : float
        Angle of the ray direction, in RADIANS.
    steps : (m,) ndarray
        Physical distances along the ray to test, in increasing order.
    spacing_2D : (2,) ndarray
        Physical voxel size in x and y, used to convert the physical
        direction/steps into index-space offsets.

    RETURNS
    -------
    dist : (n,) ndarray
        Physical distance to the first solid voxel found along `theta`, per
        candidate, or np.inf where none was found within `steps`.

    NOTE
    ----
    Ray positions are rounded to the nearest voxel index, so this is an
    approximate (grid-quantized) ray cast.
    """
    n = len(candidates)
    dist = np.full(n, np.inf)
    found = np.zeros(n, dtype=bool)

    # physical direction -> index-space step per unit of `s`
    di = np.cos(theta) / spacing_2D[0]
    dj = np.sin(theta) / spacing_2D[1]

    for s in steps:
        # advance every candidate to distance `s` along theta in one shot
        ii = np.round(candidates[:, 0] + s * di).astype(int)
        jj = np.round(candidates[:, 1] + s * dj).astype(int)
        # detect which candidates are still in bounds of the mask (some may have marched off the edge)
        in_bounds = (ii >= 0) & (ii < mask.shape[0]) & (jj >= 0) & (jj < mask.shape[1])
        # detect which candidates hit a solid voxel in the mask (only check those still in bounds)
        hit = np.zeros(n, dtype=bool)
        hit[in_bounds] = mask[ii[in_bounds], jj[in_bounds]] == 1    #if the candidate is in bounds(ii[in_bounds], jj[in_bounds]), check if it hits a solid voxel in the mask  (==1)
        # so hit is just a mask of which candidates hit a solid voxel in the mask at this step
        newly_found = hit & ~found
        dist[newly_found] = s       # record distance only the first time each candidate hits
        found = found | hit         # OR operator to update the found array with the newly found candidates
        if found.all():
            break  # every candidate already has an answer, no need to keep marching

    return dist


# =====================================================================
# 3) _detect_bridges
# =====================================================================
def _detect_bridges(support_mask: np.ndarray,
                     candidates: np.ndarray,
                     spacing_2D: np.ndarray,
                     max_span: float,
                     n_angles: int = 8) -> np.ndarray:
    """
    ============================================================================
    3) _DETECT_BRIDGES
    Tests each candidate (overhang) voxel for a bridge configuration: support
    on two roughly opposite sides, within a combined distance of `max_span`,
    along at least one tested direction. This distinguishes a two-sided
    printable bridge from a one-sided cantilevered overhang, which a plain
    (omnidirectional) distance transform cannot do.
    ============================================================================

    PARAMETERS
    ----------
    support_mask : (nx, ny) ndarray
        Binary mask of the previous (already-cleaned) slice to check support
        against.
    candidates : (n, 2) ndarray
        Row/col indices of the suspicious/overhang voxels to test.
    spacing_2D : (2,) ndarray
        Physical voxel size in x and y.
    max_span : float
        Maximum total bridgeable gap, in physical units. Rays are cast out
        to max_span / 2 in each direction.
    n_angles : int, optional
        Number of directions spanning [0, 180) degrees to test (each is
        checked together with its opposite), default = 8.

    RETURNS
    -------
    is_bridge : (n,) ndarray of bool
        True where the candidate voxel found support within `max_span` total
        (split across both sides) along at least one tested direction.
    """
    half_span = max_span / 2.0
    step_size = min(spacing_2D)
    n_steps = max(int(np.ceil(half_span / step_size)), 1)
    steps = np.arange(1, n_steps + 1) * step_size

    is_bridge = np.zeros(len(candidates), dtype=bool)
    thetas = np.linspace(0, np.pi, n_angles, endpoint=False)  # radians

    # one clear loop over directions ("for each direction, check everyone at once") --
    # the per-candidate work inside is vectorized rather than a second Python loop.
    for theta in thetas:
        # the opposite side of the SAME line through each candidate is
        # theta + pi, not -theta (which mirrors across the x-axis instead
        # of reversing the direction) -- d_plus/d_minus must be measured on
        # one straight line, not mixed across angles.
        d_plus = _ray_distance(support_mask, candidates, theta, steps, spacing_2D)
        d_minus = _ray_distance(support_mask, candidates, theta + np.pi, steps, spacing_2D)
        span = d_plus + d_minus
        is_bridge |= np.isfinite(span) & (span <= max_span)
        if is_bridge.all():
            break  # every candidate already confirmed as a bridge, no need to test more angles

    return is_bridge

# =====================================================================
# 4) _get_rotation_matrix_from_new_z_direction
# =====================================================================
def _get_rotation_matrix_from_new_z_direction(build_direction: np.ndarray,
                                            z_reference: np.ndarray = (0.0, 0.0, 1.0)) -> np.ndarray:
    """
    ============================================================================
    4) _GET_ROTATION_MATRIX_FROM_NEW_Z_DIRECTION
    Returns the rotation matrix R such that R @ z_reference == build_direction
    (both normalized first). Used to describe a candidate print orientation
    as "the axis that should become the new build (Z) direction".
    ============================================================================

    PARAMETERS
    ----------
    build_direction : (3,) array-like
        Target direction, in the object's original coordinate frame.
    z_reference : (3,) array-like, optional
        define which axis is the z direction: The axis being mapped onto build_direction (default = +Z).

    RETURNS
    -------
    R : (3, 3) ndarray
        Rotation matrix with R @ z_reference == build_direction / ||build_direction||.

    EXAMPLE
    -------
    >>> R = _get_rotation_matrix_from_new_z_direction([1, 0, 0])  # maps +Z onto +X
    """
    from scipy.spatial.transform import Rotation
    # make sure that the input vectors are normalized
    build_direction = np.asarray(build_direction, dtype=float)
    build_direction = build_direction / np.linalg.norm(build_direction)
    # make sure that the reference vector is normalized
    z_reference = np.asarray(z_reference, dtype=float)
    z_reference = z_reference / np.linalg.norm(z_reference)
    # use scipy's Rotation.align_vectors to find the rotation that aligns z_reference with build_direction
    rot, _ = Rotation.align_vectors([build_direction], [z_reference])
    # note that the two other directions are not uniquely defined
    return rot.as_matrix()

# =====================================================================
# 5) _reorient_voxel_grid
# =====================================================================
def _reorient_voxel_grid(voxel_grid: np.ndarray,
                         x: np.ndarray,
                         y: np.ndarray,
                         z: np.ndarray,
                         rotation: np.ndarray,
                         grid_sample_factor: float = 1.0) :
    """
    ============================================================================
    5) _REORIENT_VOXEL_GRID
    Reorients a 3D binary voxel grid according to a given rotation matrix.
    The reorientation is done by rotating the physical 3D coordinates of the
    solid voxels and then re-rasterizing onto a fresh regular grid.
    ============================================================================

    PARAMETERS
    ----------
    voxel_grid : (nx, ny, nz) ndarray
        3D binary voxel grid (0 = empty, 1 = solid).
    x, y, z : 1D ndarray
        Voxel coordinates along each axis (used to get the voxel pitch and
        origin). Must be evenly spaced.
    rotation : (3, 3) ndarray
        Rotation matrix applied to the object: new_point = rotation @ old_point.
        After this transform, the new +Z axis is the build direction to test.
        calculate it with _get_rotation_matrix_from_new_z_direction().

    RETURNS
    -------
    new_grid : (nx', ny', nz') ndarray
        Reoriented binary voxel grid, cropped to the bounding box of the
        rotated solid voxels (empty padding from the original grid is not
        preserved).
    new_x, new_y, new_z : 1D ndarray
        Coordinate arrays for new_grid. Pitch is equal on all three axes
        (see NOTE), so voxels stay cubic.

    RAISES
    ------
    ValueError
        If voxel_grid is not 3D, or rotation is not a 3x3 matrix.

    NOTE
    ----
    Re-rasterizing onto a fixed-pitch grid is still a voxelization step, so
    very thin features can shift by up to half a voxel or, in the worst
    case, land on the same cell as a neighboring feature -- the same
    quantization trade-off as the original voxelization, not something this
    introduces on top of it.

    version 2 (previously used, see git history) tried to cap voxel growth
    by fixing new_shape to voxel_grid.shape and solving for a spacing PER
    AXIS. That deforms the object: rotating an anisotropic bounding box
    changes each axis's extent by a different amount, so forcing a fixed
    voxel count per axis forces a different pitch per axis too -- voxels
    stop being cubes, and the shape stretches/squishes non-uniformly
    (measured: a 30deg rotation on one test box gave a Z pitch 2.8x the
    X/Y pitch). version 3 (below, active) fixes this by using ONE voxel
    size for all three axes -- sized so the TOTAL voxel count lands near
    the ORIGINAL grid's voxel count (voxel_grid.size) -- so it still
    bounds the voxel-count explosion, but can't deform the shape since
    there's only ever one voxel size in play.

    EXAMPLE
    -------
    >>> new_grid, nx_, ny_, nz_ = _reorient_voxel_grid(voxel_grid, x, y, z, R)
    """
    # ===== check inputs =====
    if voxel_grid.ndim != 3:
        raise ValueError("voxel_grid must be a 3D numpy array")

    rotation = np.asarray(rotation, dtype=float)
    if rotation.shape != (3, 3):
        raise ValueError("rotation must be a 3x3 matrix")

    if len(x) != voxel_grid.shape[0] or len(y) != voxel_grid.shape[1] or len(z) != voxel_grid.shape[2]:
        raise ValueError("x, y, and z must match the dimensions of voxel_grid")

    #===== compute position of points after rotation =====
    # get the voxel spacing and origin from the input coordinate arrays
    spacing = np.array([x[1] - x[0], y[1] - y[0], z[1] - z[0]])
    origin = np.array([x[0], y[0], z[0]])

    # sparse solid-voxel coordinates only -- this is what keeps the cost down
    idx = np.argwhere(voxel_grid == 1)
    if idx.size == 0:
        logger.debug("_reorient_voxel_grid: no solid voxels in input, returning an empty grid.")
        return (np.zeros((1, 1, 1), dtype=voxel_grid.dtype),
                np.array([0.0]), np.array([0.0]), np.array([0.0]))
    # get the physical coordinates of the solid voxel centers
    pts = origin + idx * spacing

    # O(M) matrix multiply -- exact, no interpolation, no aliasing
    # equation for each point in pts: pts_rot[i] = rotation @ pts[i]
    # where pts is a (3,1) vector and rotation is a (3,3) matrix
    # however, our pts is a (1,3) array, so we have to transpose it to (3,1) before multiplying by the rotation matrix, and then transpose it back to (1,3) after the multiplication
    # so the final equation is: pts_rot.T = (rotation @ pts.T)
    #                           pts_rot = (rotation @ pts.T).T
    # since (A*B).T = B.T * A.T
    #                           pts_rot = pts @ rotation.T
    # and we parallelize by expanding pts to (N,3) and using the @ operator for matrix multiplication in numpy.
    # thus pts_rot is a (N,3).
    pts_rot = pts @ rotation.T

    # ===== compute new grid shape and coordinates =====
    mins = pts_rot.min(axis=0)  #min value of each direction
    maxs = pts_rot.max(axis=0)  #max value of each direction
    extents = maxs - mins  # (Lx, Ly, Lz) physical size of the rotated bounding box

    #--- version 1: keep the same voxel size as the original grid ----
    # shape is the number of voxels in each direction, rounded up to the first integer above (ceiling),
    # plus 1 to include the last voxel.
    # this allows to keep the same voxel size as the original grid, but it results most often in more voxels
    # in the new grid than in the original grid, because the rotated solid voxels are not aligned with the original grid.
    """new_shape = np.ceil((maxs - mins) / spacing).astype(int) + 1

    new_x = mins[0] + np.arange(new_shape[0]) * spacing[0]
    new_y = mins[1] + np.arange(new_shape[1]) * spacing[1]
    new_z = mins[2] + np.arange(new_shape[2]) * spacing[2]"""

    # ----- version 2 (DEFORMS the shape -- kept only for reference, see NOTE above) -----
    # new_shape = np.array(voxel_grid.shape)
    # new_spacing = (maxs - mins) / (new_shape - 1)   # <- a DIFFERENT spacing per axis: this is the bug
    #
    # new_x = mins[0] + np.arange(new_shape[0]) * new_spacing[0]
    # new_y = mins[1] + np.arange(new_shape[1]) * new_spacing[1]
    # new_z = mins[2] + np.arange(new_shape[2]) * new_spacing[2]

    # ----- version 3: same goal as version 2 (bound voxel growth), without deforming the shape -----
    # use ONE voxel size for all three axes (so voxels stay cubes), 
    # sized so the TOTAL # voxel count lands near the original number
    # It is never made finer than the original pitch.
    new_volume = extents[0] * extents[1] * extents[2] * grid_sample_factor       # volume of the rotated bounding box
    new_voxel_size = (new_volume / voxel_grid.size) ** (1 / 3) #size the new voxels need to be so that the total number of voxels is the same as the original grid
    new_voxel_size = max(new_voxel_size, spacing.mean())  # never finer than the original pitch, only coarser if needed

    # calcluate the shape of the new grid based on the new voxel size and the extents of the rotated bounding box
    new_shape = np.ceil(extents / new_voxel_size).astype(int) + 1
    # calculate the new coordinates of the new grid based on the new voxel size and the minimum coordinates of the rotated bounding box
    new_x = mins[0] + np.arange(new_shape[0]) * new_voxel_size
    new_y = mins[1] + np.arange(new_shape[1]) * new_voxel_size
    new_z = mins[2] + np.arange(new_shape[2]) * new_voxel_size

    # now create a new grid in this new coordinate system that is EMPTY
    new_grid = np.zeros(new_shape, dtype=voxel_grid.dtype)
    # find in this new grid the indices of the rotated solid voxels
    new_idx = np.round((pts_rot - mins) / new_voxel_size).astype(int)
    new_idx = np.clip(new_idx, 0, new_shape - 1)  # guard against fp edge rounding
    # fill in the new grid with solid voxels at the rotated positions
    new_grid[new_idx[:, 0], new_idx[:, 1], new_idx[:, 2]] = 1

    logger.debug(f"_reorient_voxel_grid: {idx.shape[0]} solid voxels rotated, "
                 f"new grid shape {tuple(new_shape)}, pitch={new_voxel_size:.4f}.")
    return new_grid, new_x, new_y, new_z

# =====================================================================
# 6) find_optimal_orientation
# =====================================================================
def find_optimal_orientation(voxel_grid: np.ndarray,
                         x: np.ndarray,
                         y: np.ndarray,
                         z: np.ndarray,
                         n:int = 20, 
                         overhang_angle: float = 45,
                         bridge_size:float =10, 
                         grid_sample_factor: float = 1.0) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    ============================================================================
    6) FIND_OPTIMAL_ORIENTATION
    Searches for the build orientation that minimizes overhangs. Candidate
    build directions are sampled roughly evenly over the sphere (golden-angle
    spiral, `n` directions). For each candidate, voxel_grid is reoriented so
    that direction becomes the new +Z (build) axis via
    _get_rotation_matrix_from_new_z_direction + _reorient_voxel_grid, then
    scored with detect_overhangs. The orientation with the lowest score is
    reoriented and re-classified once more and returned.
    ============================================================================

    PARAMETERS
    ----------
    voxel_grid : (nx, ny, nz) ndarray
        3D binary voxel grid (0 = empty, 1 = solid).
    x, y, z : 1D ndarray
        Voxel coordinates along each axis (used to get the voxel pitch and
        origin). Must be evenly spaced.
    n : int, optional
        Number of candidate build directions to sample over the sphere
        (default = 20).
    overhang_angle : float, optional
        Overhang angle threshold in degrees, passed through to
        detect_overhangs as `angle` (default = 45).
    bridge_size : float, optional
        Maximum bridgeable span, passed through to detect_overhangs as
        `bridge` (default = 10).

    RETURNS
    -------
    overhang_grid : (nx', ny', nz') ndarray
        The detect_overhangs output (0/1/2/3 labels, see detect_overhangs)
        for the best-scoring orientation found, NOT the plain reoriented
        solid grid.
    new_x, new_y, new_z : 1D ndarray
        Coordinate arrays matching overhang_grid, for the reoriented (best)
        orientation. The winning direction vector / rotation matrix itself
        is not returned.

    NOTE
    ----
    Each candidate's score is np.sum(overhangs == 2) + np.sum(overhangs == 3),
    i.e. true overhangs and bridges are weighted equally.
    """
    # ===== check inputs =====
    if voxel_grid.ndim != 3:
        raise ValueError("voxel_grid must be a 3D numpy array")
    if x.ndim != 1 or y.ndim != 1 or z.ndim != 1:
        raise ValueError("x, y, and z must be 1D numpy arrays")
    if len(x) != voxel_grid.shape[0] or len(y) != voxel_grid.shape[1] or len(z) != voxel_grid.shape[2]:
        raise ValueError("x, y, and z must match the dimensions of voxel_grid")
    
    # ==== re-position the voxel grid so that its center is at the origin (0,0,0) ====
    x = x - np.mean(x)
    y = y - np.mean(y)
    z = z - np.mean(z)
    
    # ===== define z directions to test =====
    directions = []
    golden_angle = np.pi * (3 - np.sqrt(5))  # ~137.5 degrees, in radians
    # it will spiral around the z-axis, n/3 times

    for i in range(n):
        z_temp = 1 - (2 * i + 1) / n           # evenly spaced from +1 to -1
        radius = np.sqrt(1 - z_temp * z_temp)      # circle radius at that height
        theta = golden_angle * i         # spiral around as height decreases
        x_temp = radius * np.cos(theta)
        y_temp = radius * np.sin(theta)
        directions.append([x_temp, y_temp, z_temp])

    # ===== calculate overhangs for each direction =====
    scores = np.zeros(len(directions)) + np.inf  # initialize scores to infinity
    for idx, direction in enumerate(directions):
        Rotation_matrix =_get_rotation_matrix_from_new_z_direction(direction)
        Rotated_structure, new_x, new_y, new_z = _reorient_voxel_grid(voxel_grid, x, y, z, Rotation_matrix, grid_sample_factor=grid_sample_factor)
        overhangs = detect_overhangs(Rotated_structure, new_x, new_y, new_z, angle=overhang_angle, bridge=bridge_size, add_support_voxels=True)
        scores[idx] = np.sum(overhangs == 2) + np.sum(overhangs == 3)  # count overhangs and bridges
        logger.info(f"Tested direction {idx+1}/{len(directions)}. Overhangs: {np.sum(overhangs == 2)}, Bridges: {np.sum(overhangs == 3)}, Total: {scores[idx]}")

    best_angle = directions[np.argmin(scores)]
    Rotation_matrix =_get_rotation_matrix_from_new_z_direction(best_angle)
    Rotated_structure, new_x, new_y, new_z = _reorient_voxel_grid(voxel_grid, x, y, z, Rotation_matrix, grid_sample_factor=grid_sample_factor)
    Rotated_structure = detect_overhangs(Rotated_structure, new_x, new_y, new_z, angle=overhang_angle, bridge=bridge_size, add_support_voxels=True)
    return Rotated_structure, new_x, new_y, new_z
import numpy as np
from collections import defaultdict
import trimesh # type: ignore
import open3d as o3d # type: ignore
from .logger import logger # type: ignore
from stl import mesh as stl_mesh # type: ignore
from skimage import measure # type: ignore
import pymeshfix # type: ignore


"""
#=====================================================================================================================
0 - (reserved)
1 - keep_largest_connected_component
2 - simplify_mesh
3 - calculate_triangle_areas
4 - export_as_STL
5 - mesh_from_matrix
6 - check_mesh_validity
7 - fix_mesh
#=====================================================================================================================
"""

# =====================================================================
# 1) keep_largest_connected_component
# =====================================================================
def keep_largest_connected_component(verts, faces):
    """
    ============================================================================
    1) KEEP_LARGEST_CONNECTED_COMPONENT
    Extracts and returns only the largest connected component in a triangular mesh.
    ============================================================================

    PARAMETERS
    ----------
    verts : (N, 3) ndarray
        Vertex coordinates.
    faces : (M, 3) ndarray
        Triangular face indices.

    RETURNS
    -------
    largest_vertices : (K, 3) ndarray
        Vertices belonging to the largest connected region.
    largest_faces : (L, 3) ndarray
        Faces belonging to the largest connected region.

    EXAMPLE
    -------
    >>> v2, f2 = keep_largest_connected_component(vertices, faces)
    >>> print(v2.shape, f2.shape)
    """
    logger.info("Extracting largest connected component…")

    # ------------------------------------------------------------------
    # Validate inputs
    # ------------------------------------------------------------------
    if verts is None or faces is None:
        logger.error("Input verts or faces is None.")
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=int)

    if len(faces) == 0:
        logger.warning("Mesh has zero faces — nothing to extract.")
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=int)

    logger.debug(f"Input mesh: {len(verts)} vertices, {len(faces)} faces")

    # ------------------------------------------------------------------
    # find largest component
    # ------------------------------------------------------------------
    try:
        mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
        components = mesh.split(only_watertight=False)
    except Exception as e:
        logger.error(f"Failed to split mesh into components: {e}", exc_info=True)
        return verts, faces

    if len(components) == 0:
        logger.warning("No connected components found in mesh.")
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=int)

    # ------------------------------------------------------------------
    # result
    # ------------------------------------------------------------------
    largest = max(components, key=lambda m: len(m.faces))
    logger.info(
        f"Selected largest component: {len(largest.vertices)} vertices, "
        f"{len(largest.faces)} faces"
    )

    return largest.vertices, largest.faces



# =====================================================================
# 2) simplify_mesh
# =====================================================================
def simplify_mesh(faces, verts, target=100000):
    """
    ============================================================================
    2) SIMPLIFY_MESH
    Iteratively decimates a mesh using Open3D quadric decimation until a target
    number of faces is reached.
    ============================================================================

    PARAMETERS
    ----------
    faces : (M, 3) ndarray
        Triangular faces.
    verts : (N, 3) ndarray
        Vertex coordinates.
    target : int, optional
        Desired final number of faces (default = 100000).

    RETURNS
    -------
    faces_simplified : (T, 3) ndarray
        Simplified faces.
    verts_simplified : (S, 3) ndarray
        Simplified vertices.

    EXAMPLE
    -------
    >>> f2, v2 = simplify_mesh(faces, verts, target=50000)
    >>> print(len(f2))
    """
    if faces is None or verts is None:
        logger.error("simplify_mesh(): faces or verts is None.")
        return faces, verts

    original = len(faces)
    current = original

    logger.info(f"Simplifying mesh: {original} faces → target {target}")

    if original == 0:
        logger.warning("Mesh has zero faces — skipping simplification.")
        return faces, verts

    i = 0
    while current > target:
        i += 1
        current = max(int(current * 0.5), target)

        logger.debug(f"[Step {i}] Target face count: {current}")

        try:
            tri = o3d.geometry.TriangleMesh(
                o3d.utility.Vector3dVector(verts),
                o3d.utility.Vector3iVector(faces)
            )

            tri = tri.simplify_quadric_decimation(
                target_number_of_triangles=current
            )

            verts = np.asarray(tri.vertices)
            faces = np.asarray(tri.triangles)

        except Exception as e:
            logger.error(f"Mesh simplification failed at step {i}: {e}", exc_info=True)
            break
    
    # ------------------------------------------------------------------
    # Fix mesh (normals, non-manifold edges)
    # ------------------------------------------------------------------

    logger.info(f"Mesh simplification complete → {len(faces)} faces remain.")

    return faces, verts


#=====================================================================
#3) TRIANGLE_AREAS
#=====================================================================
def calculate_triangle_areas(verts: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """
    ============================================================================
    3) TRIANGLE_AREAS
    Computes area for every triangular face in a mesh.
    ============================================================================

    PARAMETERS
    ----------
    verts : (N, 3) ndarray
        Vertex positions.
    faces : (M, 3) ndarray
        Triangular connectivity.

    RETURNS
    -------
    areas : (M,) ndarray
        Area of each triangle.

    EXAMPLE
    -------
    >>> areas = triangle_areas(verts, faces)
    >>> print("Min area:", areas.min())
    """
    logger.info("Calculating triangle areas…")

    if verts is None or faces is None:
        logger.error("calculate_triangle_areas(): verts or faces is None.")
        return np.zeros(0)

    if len(faces) == 0:
        logger.warning("calculate_triangle_areas(): no faces provided.")
        return np.zeros(0)

    logger.debug(f"Computing areas for {len(faces)} faces.")

    try:
        V = verts.astype(np.float64, copy=False)
        v0 = V[faces[:, 0]]
        v1 = V[faces[:, 1]]
        v2 = V[faces[:, 2]]

        cross_prod = np.cross(v1 - v0, v2 - v0)
        areas = 0.5 * np.linalg.norm(cross_prod, axis=1)

        logger.info("Triangle area computation complete.")
        logger.debug(f"Area stats — min: {areas.min()}, max: {areas.max()}")

        return areas

    except Exception as e:
        logger.error(f"Error computing triangle areas: {e}", exc_info=True)
        return np.zeros(0)



#=====================================================================
#4) export_as_STL
#=====================================================================
def export_as_STL(verts: np.ndarray, faces: np.ndarray, path: str):
    """
    ============================================================================
    4) EXPORT_AS_STL
    Exports a triangle mesh defined by (verts, faces) to an STL file.
    ============================================================================

    PARAMETERS
    ----------
    verts : (N, 3) ndarray
        Vertex coordinates.
    faces : (M, 3) ndarray
        Triangle face connectivity.
    path : str
        Output STL file path.

    RETURNS
    -------
    None
    """
    logger.info(f"Exporting STL → {path}")

    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]

    # ------------------------------------------------------------------
    # Validate inputs
    # ------------------------------------------------------------------
    if verts is None or faces is None:
        logger.error("export_as_STL(): verts or faces is None.")
        return

    if len(faces) == 0:
        logger.warning("export_as_STL(): No faces to export; STL will be empty.")
        return

    # -------------------------
    # Build STL mesh (vectorized)
    # -------------------------
    try:
        stl_obj = stl_mesh.Mesh(np.zeros(faces.shape[0], dtype=stl_mesh.Mesh.dtype))
        stl_obj.vectors[:, 0, :] = v0
        stl_obj.vectors[:, 1, :] = v1
        stl_obj.vectors[:, 2, :] = v2
    except Exception as e:
        logger.error(f"STL mesh construction failed: {e}", exc_info=True)
        return

    # ------------------------------------------------------------------
    # Save file
    # ------------------------------------------------------------------
    try:
        stl_obj.save(path)
        logger.info(f"STL successfully saved: {path}")
    except Exception as e:
        logger.error(f"Failed to save STL file '{path}': {e}", exc_info=True)





#=====================================================================
#5) mesh_from_matrix
#=====================================================================
def mesh_from_matrix(
    matrix: np.ndarray,
    iso_level: float,
    algo_step_size: int,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    pad_width: int = 5,
    pad_val: float = -1
    ):
    """
    ============================================================================
    5) MESH_FROM_MATRIX
    Extracts an isosurface mesh (verts, faces) from a 3D scalar field using
    marching cubes. Optionally pads the volume with a constant value to help 
    close surfaces touching the boundary).
    ============================================================================
    
    PARAMETERS
    ----------
    matrix : (nx, ny, nz) ndarray
        Scalar field values sampled on a regular grid.
    iso_level : float
        Isosurface level (marching cubes "level").
    spacing : tuple(float, float, float)
        Voxel spacing (dx, dy, dz) in physical units.
    algo_step_size : int
        Marching cubes step size (larger = faster, less detail).
    x, y, z : float, optional
        Physical coordinate of voxel (0,0,0) in the *un-padded* matrix.
    pad_width : int, optional
        Number of voxels to pad on each face of the volume.
    pad_val : float or None, optional
        Constant padding value. If None, automatically chosen to be safely above
        iso_level relative to the data range (encourages "caps" on boundaries).
        If your "solid" is on the other side of the iso_level, you may want to
        pass a value safely below iso_level instead.

    RETURNS
    -------
    verts : (N, 3) ndarray
        Vertex coordinates in physical units.
    faces : (M, 3) ndarray
        Triangle connectivity (indices into verts).
    """
    logger.info("Extracting isosurface mesh from matrix.")

    # ------------------------------------------------------------------
    # Pad volume to help close boundary openings ("caps")
    # ------------------------------------------------------------------
    try:
        v_padded = np.pad(matrix, pad_width=pad_width, mode="constant", constant_values=pad_val)
    except Exception as e:
        logger.error(f"np.pad failed: {e}", exc_info=True)
        return None, None

    # ------------------------------------------------------------------
    # Extract isosurface
    # ------------------------------------------------------------------
    spacing = (np.abs(x[1,0,0]-x[0,0,0]),
            np.abs(y[0,1,0]-y[0,0,0]),
            np.abs(z[0,0,1]-z[0,0,0]))  

    try:
        verts, faces, normals, values = measure.marching_cubes(
            v_padded,
            level=iso_level,
            spacing=spacing,
            step_size=int(algo_step_size),
            allow_degenerate=False,
        )
    except Exception as e:
        logger.error(f"marching_cubes failed: {e}", exc_info=True)
        return None, None

    # ------------------------------------------------------------------
    # Translate vertices to the physical coordinate system of the unpadded grid
    # ------------------------------------------------------------------
    x_origin = x[0,0,0]
    y_origin = y[0,0,0]
    z_origin = z[0,0,0]

    origin = (x_origin - pad_width * spacing[0], y_origin - pad_width * spacing[1], z_origin - pad_width * spacing[2])

    try:
        verts[:, 0] = verts[:,0] + origin[0]
        verts[:, 1] = verts[:,1] + origin[1]
        verts[:, 2] = verts[:,2] + origin[2]
    except Exception as e:
        logger.error(f"vertex translation failed: {e}", exc_info=True)
        return None, None

    logger.info(f"Extracted mesh with {len(verts)} verts and {len(faces)} faces.")

    # ------------------------------------------------------------------
    # Fix mesh (normals, non-manifold edges)
    # ------------------------------------------------------------------
    #verts, faces = fix_mesh(verts, faces)

    # ------------------------------------------------------------------
    # result
    # ------------------------------------------------------------------
    return verts, faces



#=====================================================================
#6) check_mesh_validity
#=====================================================================
def check_mesh_validity(verts: np.ndarray, faces: np.ndarray):
    """
    ============================================================================
    6) CHECK_MESH_VALIDITY
    Performs basic topological and geometric validity checks on a triangle mesh
    using trimesh.
    ============================================================================
    
    PARAMETERS
    ----------
    verts : (N, 3) ndarray
        Vertex coordinates.
    faces : (M, 3) ndarray
        Triangle face connectivity.

    RETURNS
    -------
    info : dict
        Dictionary with mesh validity indicators.
    """

    logger.info("check_mesh_validity(): Checking mesh validity.")

    # ------------------------------------------------------------------
    # Build mesh
    # ------------------------------------------------------------------
    try:
        m = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    except Exception as e:
        logger.error(f"check_mesh_validity(): Mesh construction failed: {e}", exc_info=True)
        return None

    edge_counts = np.bincount(m.edges_unique_inverse)

    # ------------------------------------------------------------------
    # Collect validity metrics
    # ------------------------------------------------------------------
    info = {
        "watertight": m.is_watertight,
        "winding_consistent": m.is_winding_consistent,
        "is_volume": m.is_volume,
        "boundary_edges": int(np.sum(edge_counts == 1)),
        "nonmanifold_edges": int(np.sum(edge_counts > 2)),
        "self_intersecting": (
            m.is_self_intersecting
            if hasattr(m, "is_self_intersecting")
            else False  # not supported in this trimesh version
        ),
    }
    if m.is_volume and m.is_watertight and m.is_winding_consistent:
        logger.info(f"check_mesh_validity(): {info}")
    else:
        logger.warning(f"check_mesh_validity(): Mesh is not a valid volume. Details: {info}")
    return info


#=====================================================================
#7) fix_mesh
#=====================================================================
def fix_mesh(verts: np.ndarray, faces: np.ndarray, recursion_depth: int = 5):
    recursion_depth -= 1
    # build trimesh object without automatic processing
    m = trimesh.Trimesh(vertices=verts, faces=faces, process=False)

    # ------------------------------------------------------------------
    # Step A: Construct a trimesh object and attempt to fix normals.
    # - Call `fix_normals()` so face orientations are consistent where
    #   possible; this helps later checks like `is_volume`.
    # ------------------------------------------------------------------
    # Call repair functions defensively: some trimesh versions modify the
    # mesh in-place, others may return a modified mesh or arrays. Handle
    # both cases to stay compatible across versions.
    try:
        res = trimesh.repair.fix_normals(m)
        if isinstance(res, trimesh.Trimesh):
            m = res
    except Exception:
        pass

    try:
        res = trimesh.repair.remove_duplicate_faces(m)
        if isinstance(res, trimesh.Trimesh):
            m = res
        elif isinstance(res, (list, tuple, np.ndarray)):
            # some versions return faces or face-index arrays
            try:
                m.faces = np.asarray(res)
            except Exception:
                pass
    except Exception:
        pass

    try:
        res = trimesh.repair.remove_degenerate_faces(m)
        if isinstance(res, trimesh.Trimesh):
            m = res
        elif isinstance(res, (list, tuple, np.ndarray)):
            try:
                m.faces = np.asarray(res)
            except Exception:
                pass
    except Exception:
        pass

    try:
        if hasattr(trimesh.repair, "fill_holes"):
            res = trimesh.repair.fill_holes(m)
            if isinstance(res, trimesh.Trimesh):
                m = res
    except Exception:
        pass

    verts = m.vertices
    faces = m.faces

    if _is_mesh_fixed(verts, faces):
        # mesh is already valid after normal fixing; return early
        logger.info("mesh is already valid after normal fixing.")
        return verts, faces

    
    # ------------------------------------------------------------------
    # Step B: use pymeshfix to attempt to fix non-manifold edges and other common issues.
    # ------------------------------------------------------------------
    mf = pymeshfix.MeshFix(verts, faces)
    mf.repair()        # modifies and repairs in-place
    verts, faces = mf.verts, mf.faces   

    if _is_mesh_fixed(verts, faces):
        logger.info("fix_mesh(): mesh successfully fixed and is now valid.")

    # ------------------------------------------------------------------
    # Step C: use mesh simplifier to attempt to fix any remaining issues
    # ------------------------------------------------------------------
    verts, faces = simplify_mesh(verts, faces, target=int(len(faces)*0.95))

    if _is_mesh_fixed(verts, faces):
        logger.info("fix_mesh(): mesh successfully fixed and is now valid after simplification.")

    else:
        logger.warning("fix_mesh(): mesh is still invalid after all fixing attempts.")
        if recursion_depth > 0:
            logger.info(f"fix_mesh(): retrying fix_mesh() (recursion depth remaining: {recursion_depth})")
            return fix_mesh(verts, faces, recursion_depth=recursion_depth)

    # ------------------------------------------------------------------
    # Finalize: report resulting mesh size and return the arrays
    # ------------------------------------------------------------------
    logger.info(f"fix_mesh(): returning mesh with {len(faces)} faces and {len(verts)} verts")
    return verts, faces


def _is_mesh_fixed(verts: np.ndarray, faces: np.ndarray) -> bool:
    """
    Helper function to check if a mesh is valid (watertight, manifold, etc.)
    Used for internal testing purposes.
    """
    info = check_mesh_validity(verts, faces)
    if info is None:
        return False
    return info["watertight"] and info["winding_consistent"] and info["nonmanifold_edges"] == 0 and not info["self_intersecting"]==0
import numpy as np
import trimesh
import open3d as o3d
from .logger import logger
from stl import mesh as stl_mesh
from skimage import measure


"""
#=====================================================================================================================
0 - (reserved)
1 - keep_largest_connected_component
2 - simplify_mesh
3 - TRIANGLE_AREAS
4 - export_as_STL
5 - mesh_from_matrix
6 - check_mesh_validity
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
    pad_val: float = None,
    ):
    logger.info("mesh_from_matrix(): Extracting isosurface mesh from matrix.")

    # ------------------------------------------------------------------
    # Pad volume to help close boundary openings ("caps")
    # ------------------------------------------------------------------
    try:
        v_padded = np.pad(matrix, pad_width=pad_width, mode="constant", constant_values=pad_val)
    except Exception as e:
        logger.error(f"mesh_from_matrix(): np.pad failed: {e}", exc_info=True)
        return None, None

    # ------------------------------------------------------------------
    # Extract isosurface
    # ------------------------------------------------------------------
    spacing = (np.abs(x[1,0,0]-x[0,0,0]),
            np.abs(y[0,1,0]-y[0,0,0]),
            np.abs(z[0,0,1]-z[0,0,0]))  
    
    logger.debug(f"mesh_from_matrix(): spacing = {spacing}")
    try:
        verts, faces, normals, values = measure.marching_cubes(
            v_padded,
            level=iso_level,
            spacing=spacing,
            step_size=int(algo_step_size),
            allow_degenerate=False)
    except Exception as e:
        logger.error(f"mesh_from_matrix(): marching_cubes failed: {e}", exc_info=True)
        return None, None

    logger.debug(f"mesh_from_matrix(): raw verts bbox: [{np.min(verts[:, 0]):.3f}, {np.max(verts[:, 0]):.3f}] x [{np.min(verts[:, 1]):.3f}, {np.max(verts[:, 1]):.3f}] x [{np.min(verts[:, 2]):.3f}, {np.max(verts[:, 2]):.3f}]")
    
    # ------------------------------------------------------------------
    # Translate vertices to the physical coordinate system of the unpadded grid
    # ------------------------------------------------------------------
    try:
        # Extract 1D axis arrays
        xa = np.asarray(x)
        ya = np.asarray(y)
        za = np.asarray(z)
        if xa.ndim > 1:
            xa = xa[:, 0, 0]
        if ya.ndim > 1:
            ya = ya[0, :, 0]
        if za.ndim > 1:
            za = za[0, 0, :]

        Nx, Ny, Nz = len(xa), len(ya), len(za)

        # Convert marching_cubes physical coords → padded fractional indices
        verts_idx_x = verts[:, 0] / float(spacing[0]) - pad_width - 0.5
        verts_idx_y = verts[:, 1] / float(spacing[1]) - pad_width - 0.5
        verts_idx_z = verts[:, 2] / float(spacing[2]) - pad_width - 0.5

        # Clamp and interpolate to physical axis coordinates
        verts_idx_x = np.clip(verts_idx_x, 0.0, Nx - 1.0)
        verts_idx_y = np.clip(verts_idx_y, 0.0, Ny - 1.0)
        verts_idx_z = np.clip(verts_idx_z, 0.0, Nz - 1.0)

        ix = np.arange(Nx)
        iy = np.arange(Ny)
        iz = np.arange(Nz)

        verts[:, 0] = np.interp(verts_idx_x, ix, xa)
        verts[:, 1] = np.interp(verts_idx_y, iy, ya)
        verts[:, 2] = np.interp(verts_idx_z, iz, za)
        
        logger.debug(f"mesh_from_matrix(): mapped verts bbox: [{verts[:, 0].min():.3f}, {verts[:, 0].max():.3f}] x [{verts[:, 1].min():.3f}, {verts[:, 1].max():.3f}] x [{verts[:, 2].min():.3f}, {verts[:, 2].max():.3f}]")
    except Exception as e:
        logger.error(f"mesh_from_matrix(): vertex translation failed: {e}", exc_info=True)
        return None, None

    logger.info(f"mesh_from_matrix(): Extracted {len(verts)} vertices, {len(faces)} faces")
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
    # Build mesh (no auto-processing)
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

    logger.info(f"check_mesh_validity(): {info}")
    return info


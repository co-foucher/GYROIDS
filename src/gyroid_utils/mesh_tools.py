import numpy as np
import trimesh
import open3d as o3d
from .logger import logger
from stl import mesh as stl_mesh


"""
#=====================================================================================================================
0 - (reserved)
1 - keep_largest_connected_component
2 - simplify_mesh
3 - TRIANGLE_AREAS
4 - export_as_STL
5 - mesh_from_matrix
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

    if verts is None or faces is None:
        logger.error("Input verts or faces is None.")
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=int)

    if len(faces) == 0:
        logger.warning("Mesh has zero faces — nothing to extract.")
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=int)

    logger.debug(f"Input mesh: {len(verts)} vertices, {len(faces)} faces")

    try:
        mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
        components = mesh.split(only_watertight=False)
    except Exception as e:
        logger.error(f"Failed to split mesh into components: {e}", exc_info=True)
        return verts, faces

    if len(components) == 0:
        logger.warning("No connected components found in mesh.")
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=int)

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

    # ------------------------------------------------------------------
    # Validate inputs
    # ------------------------------------------------------------------
    if verts is None or faces is None:
        logger.error("export_as_STL(): verts or faces is None.")
        return

    if len(faces) == 0:
        logger.warning("export_as_STL(): No faces to export; STL will be empty.")
        return

    # ------------------------------------------------------------------
    # drop degenerate triangles (zero area)
    # ------------------------------------------------------------------
    # (Keeps export clean for some viewers/meshing tools.)
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    cross = np.cross(v1 - v0, v2 - v0)
    area2 = np.einsum("ij,ij->i", cross, cross)  # proportional to area^2
    keep = area2 > 0.0

    if not np.all(keep):
        removed = int(np.size(keep) - np.count_nonzero(keep))
        logger.warning(f"export_as_STL(): Removing {removed} degenerate triangles.")
        faces = faces[keep]
        if len(faces) == 0:
            logger.error("export_as_STL(): All triangles were degenerate; nothing to export.")
            return
        v0 = v0[keep]; v1 = v1[keep]; v2 = v2[keep]

    # -------------------------
    # Build STL mesh (vectorized)
    # -------------------------
    try:
        stl_obj = stl_mesh.Mesh(np.zeros(faces.shape[0], dtype=stl_mesh.Mesh.dtype))
        stl_obj.vectors[:, 0, :] = v0
        stl_obj.vectors[:, 1, :] = v1
        stl_obj.vectors[:, 2, :] = v2
        stl_obj.update_normals()
    except Exception as e:
        logger.error(f"STL mesh construction failed: {e}", exc_info=True)
        return

    # ------------------------------------------------------------------
    # Save file
    # ------------------------------------------------------------------
    try:
        stl_obj.save(path)
        logger.info(f"STL successfully saved: {path}")
        print("sadfbgsadfbsdfbsbsgfbkjnsdlifkjnbvedslif")
    except Exception as e:
        logger.error(f"Failed to save STL file '{path}': {e}", exc_info=True)






#=====================================================================
#5) mesh_from_matrix
#=====================================================================
def mesh_from_matrix(matrix:np.ndarray, iso_level, spacing, algo_step_size, x=0, y=0, z=0):
    # --- Ensure the surface is CLOSED (emulates MATLAB's isocaps(..., "below")) ---
    # Strategy: we want the interior to be "v < iso_level" (the "below" side).
    # If the surface hits the boundary, it's open. We close it by padding a 1-voxel thick
    # frame of large POSITIVE values (>> iso_level) around v.
    # That guarantees a crossing at the boundary and thus creates "caps" there.
    pad_val = 0                                              # safely above any v near 0
    v_padded = np.pad(matrix, pad_width=5, mode='constant', constant_values=pad_val)

    # The physical origin of the padded grid is shifted by -1 voxel in each direction:
    origin = (x1.min() - dx_grid, y1.min() - dy_grid, z1.min() - dz_grid)


    # --- Extract the isosurface in one pass, scaled to physical units ---
    # Using 'spacing' returns vertices already scaled; we still need to add 'origin'.
    verts, faces, normals, values = measure.marching_cubes(
        v_padded, level=iso_level, spacing=spacing, step_size=algo_step_size, allow_degenerate=False)

    # Translate vertices so they live in the same physical coordinate system as (x,y,z)
    verts[:, 0] += origin[0]
    verts[:, 1] += origin[1]
    verts[:, 2] += origin[2]

    print(f"there are {len(faces)} faces in this model")

    return verts, faces
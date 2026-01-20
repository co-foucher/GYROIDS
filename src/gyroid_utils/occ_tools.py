import numpy as np

from OCC.Core.Poly import Poly_Triangulation
from OCC.Core.ShapeUpgrade import ShapeUpgrade_ShapeDivideAngle
from OCC.Core.ShapeAnalysis import ShapeAnalysis_Shell
from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
from OCC.Core.IFSelect import IFSelect_RetDone
from OCC.Core.STEPControl import STEPControl_Writer, STEPControl_AsIs
from OCC.Core.ShapeFix import ShapeFix_Shell
from OCC.Core.TopAbs import TopAbs_SHELL
from OCC.Core.BRep import BRep_Builder
from OCC.Core.BRepCheck import BRepCheck_Shell, BRepCheck_Analyzer
from OCC.Core.TopoDS import TopoDS_Compound, TopoDS_Shape, TopoDS_Shell
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_FACE

from OCC.Core.BRepBuilderAPI import (
    BRepBuilderAPI_MakePolygon,
    BRepBuilderAPI_MakeWire,
    BRepBuilderAPI_MakeFace,
    BRepBuilderAPI_Sewing,
    BRepBuilderAPI_MakeSolid,
)

from OCC.Core.gp import gp_Pnt
from OCC.Core.ShapeUpgrade import ShapeUpgrade_UnifySameDomain

from .mesh_tools import calculate_triangle_areas
from .logger import logger

from time import perf_counter
"""
#=====================================================================================================================
0 - (reserved)
1 - is_shell_closed
2 - count_faces
3 - make_triangle_face
4 - is_valid_shape
5 - simplify_shell
6 - make_shell_from_faces
7 - faces_to_solid
#=====================================================================================================================
"""


#=====================================================================
#1) IS_SHELL_CLOSED
#=====================================================================
def is_shell_closed(shell):
    """
    ============================================================================
    1) IS_SHELL_CLOSED
    Checks whether an OpenCascade shell is geometrically closed.
    ============================================================================

    PARAMETERS
    ----------
    shell : TopoDS_Shell

    RETURNS
    -------
    closed : bool
        True if the shell is closed.

    NOTES
    -----
    - BRepCheck_Shell returns 0 when closed.

    EXAMPLE
    -------
    >>> closed = is_shell_closed(my_shell)
    >>> print("Closed:", closed)
    """
    checker = BRepCheck_Shell(shell)
    if shell is None or shell.IsNull():
        logger.warning("is_shell_closed() called with NULL shell.")
        return False

    checker = BRepCheck_Shell(shell)
    closed = (checker.Closed() == 0)

    logger.debug(f"Shell closed check: {closed}")
    return closed


#=====================================================================
#2) count_faces
#=====================================================================
def count_faces(shape):
    """
    ============================================================================
    2) COUNT_FACES
    Counts the number of faces in an OpenCascade shape.
    ============================================================================

    PARAMETERS
    ----------
    shape : TopoDS_Shape

    RETURNS
    -------
    count : int
        Number of faces in the shape.

    EXAMPLE
    -------
    >>> n_faces = count_faces(shape)
    >>> print(n_faces)
    """
    if shape is None or shape.IsNull():
        logger.warning("count_faces() called with NULL shape.")
        return 0

    exp = TopExp_Explorer(shape, TopAbs_FACE)
    n = 0
    while exp.More():
        n += 1
        exp.Next()

    logger.debug(f"count_faces(): {n} faces found.")
    return n


#=====================================================================
#3) make_triangle_face
#=====================================================================
def make_triangle_face(p0, p1, p2):
    """
    ============================================================================
    3) MAKE_TRIANGLE_FACE
    Creates a planar OCC face from 3 vertices.
    ============================================================================

    PARAMETERS
    ----------
    p0, p1, p2 : array-like (3,)
        Coordinates of the triangle vertices.

    RETURNS
    -------
    face : TopoDS_Face
        OCC face representing the triangle.

    EXAMPLE
    -------
    >>> f = make_triangle_face(v0, v1, v2)
    """
    logger.debug(f"Creating triangle face from points: {p0}, {p1}, {p2}")

    try:
        poly = BRepBuilderAPI_MakePolygon()
        poly.Add(gp_Pnt(*map(float, p0)))
        poly.Add(gp_Pnt(*map(float, p1)))
        poly.Add(gp_Pnt(*map(float, p2)))
        poly.Close()

        wire = BRepBuilderAPI_MakeWire(poly.Wire()).Wire()
        face = BRepBuilderAPI_MakeFace(wire).Face()

        return face

    except Exception as e:
        logger.error(f"Failed to create triangle face: {e}", exc_info=True)
        raise


#=====================================================================
#4) is_valid_shape
#=====================================================================
def is_valid_shape(shape):
    """
    ============================================================================
    4) IS_VALID_SHAPE
    Uses OpenCascade's BRepCheck_Analyzer to verify whether a shape is valid.
    ============================================================================

    PARAMETERS
    ----------
    shape : TopoDS_Shape

    RETURNS
    -------
    valid : bool

    EXAMPLE
    -------
    >>> print(is_valid_shape(shape))
    """
    if shape is None or shape.IsNull():
        logger.warning("is_valid_shape(): NULL or invalid shape given.")
        return False

    ana = BRepCheck_Analyzer(shape)
    valid = ana.IsValid()

    logger.debug(f"is_valid_shape(): {valid}")
    return valid


#=====================================================================
#5) simplify_shell
#=====================================================================
def simplify_shell(shell, linear_tol=1e-6, angular_tol_deg=2):
    """
    ============================================================================
    5) SIMPLIFY_SHELL
    Simplifies an OCC shell using ShapeUpgrade_UnifySameDomain.
    ============================================================================
    
    Performs:
    ----------
    - Merges coplanar faces
    - Merges tangent-continuous faces
    - Reduces face count
    - Applies user-specified tolerances

    PARAMETERS
    ----------
    shell : TopoDS_Shape (shell)
    linear_tol : float
        Maximum allowed gap for merging.
    angular_tol_deg : float
        Maximum allowed angle between face normals.

    RETURNS
    -------
    simplified_shell : TopoDS_Shape
        The merged shell (or original shell if merging fails).

    EXAMPLE
    -------
    >>> new_shell = simplify_shell(shell, linear_tol=1e-5)
    """
    logger.info("Starting shell simplification...")

    # Validate input
    if shell is None or shell.IsNull():
        logger.warning("simplify_shell skipped: NULL input shell.")
        return shell

    if not is_valid_shape(shell):
        logger.warning("simplify_shell skipped: invalid input shell.")
        return shell

    faces_before = count_faces(shell)
    if faces_before == 0:
        logger.info("simplify_shell skipped: shell has zero faces.")
        return shell

    logger.info(
        f"Simplify shell: {faces_before} faces before merging. "
        f"(linear_tol={linear_tol}, angular_tol_deg={angular_tol_deg})"
    )

    try:
        unifier = ShapeUpgrade_UnifySameDomain(shell, True, True, True)
        unifier.SetLinearTolerance(linear_tol)
        unifier.SetAngularTolerance(np.deg2rad(angular_tol_deg))

        unifier.Build()

    except Exception as e:
        logger.error(f"simplify_shell failed during merging: {e}", exc_info=True)
        return shell

    # Extract simplified shell
    shell_simplified = unifier.Shape()

    if shell_simplified is None or shell_simplified.IsNull():
        logger.error("simplify_shell output is NULL → keeping original shell.")
        return shell

    faces_after = count_faces(shell_simplified)
    reduction = 100 * (faces_before - faces_after) / faces_before

    logger.info(
        f"Simplification result: {faces_before} → {faces_after} faces "
        f"({reduction:.1f}% reduction)"
    )

    return shell_simplified


#=====================================================================
#6) MAKE_SHELL_FROM_FACES
#=====================================================================
def make_shell_from_faces(vertices, faces_chunk, sew_tol, faces_area=None):
    """
    ============================================================================
    6) MAKE_SHELL_FROM_FACES
    Builds a sewn OCC shell from triangle faces.
    ============================================================================

    PARAMETERS
    ----------
    vertices : (N, 3) ndarray
    faces_chunk : (M, 3) ndarray
        A subset of triangle indices.
    sew_tol : float
        Sewing tolerance.
    faces_area : (M,) ndarray or None
        If provided, triangles with area < sew_tol^2 are skipped.

    RETURNS
    -------
    shell : TopoDS_Shape
        The sewn shell.
    skipped : int
        Number of triangles skipped due to filtering.

    PROCESS
    -------
    1. Build individual planar triangle faces  
    2. Optionally skip tiny triangles  
    3. Add all faces into a compound  
    4. Use BRepBuilderAPI_Sewing to merge all into one shell  

    EXAMPLE
    -------
    >>> shell, skipped = make_shell_from_faces(verts, faces, sew_tol=1e-3)
    """
    logger.info(
        f"Building shell from {len(faces_chunk)} triangles "
        f"(sew_tol={sew_tol}, area filter={'on' if faces_area is not None else 'off'})"
    )

    bb = BRep_Builder()
    comp = TopoDS_Compound()
    bb.MakeCompound(comp)

    skipped = 0

    for tri_idx, tri in enumerate(faces_chunk):

        # Optional area filter
        if faces_area is not None:
            if tri_idx < len(faces_area) and faces_area[tri_idx] < sew_tol**2:
                skipped += 1
                logger.debug(f"Skipping tiny triangle {tri_idx} (area={faces_area[tri_idx]})")
                continue

        p0, p1, p2 = vertices[tri]

        try:
            face = make_triangle_face(p0, p1, p2)
            bb.Add(comp, face)
        except Exception as e:
            logger.error(f"Failed to create triangle face {tri_idx}: {e}", exc_info=True)
            skipped += 1

    logger.info(f"Finished building faces. {skipped} triangles skipped. Sewing shell...")

    try:
        sewer = BRepBuilderAPI_Sewing()
        sewer.SetTolerance(sew_tol)
        sewer.SetNonManifoldMode(False)
        sewer.Add(comp)
        sewer.Perform()

        shell = sewer.SewedShape()

        if shell is None or shell.IsNull():
            logger.error("Sewing produced NULL shell.")
            return None, skipped

        logger.info("Shell successfully sewn.")
        return shell, skipped

    except Exception as e:
        logger.error(f"Sewing failed: {e}", exc_info=True)
        return None, skipped


#=====================================================================
#7) faces_to_solid
#=====================================================================


def faces_to_solid(vertices, faces, faces_area=None, sew_tol=1e-5):
    """
    ============================================================================
    FACES_TO_SOLID
    Convert a triangle mesh into an OpenCascade solid.
    ============================================================================
    
    PARAMETERS
    ----------
    vertices : (N, 3) ndarray
        Mesh vertices.
    faces : (M, 3) ndarray
        Triangle face connectivity.
    faces_area : (M,) ndarray or None
        Optional area array to filter tiny faces.
    sew_tol : float
        Sewing tolerance for the OCC sewing algorithm.

    RETURNS
    -------
    solid_normal : TopoDS_Shape
        Initial solid created from unsimplified shell.
    solid_simplified : TopoDS_Shape
        Final simplified solid.
    """

    logger.info("Starting faces_to_solid()")
    logger.info(f"Number of faces: {len(faces)}")
    logger.info(f"Sew tolerance: {sew_tol}")

    t_start = perf_counter()

    # -------------------------------------------------------------
    # 1) Sew triangle faces into a shell
    # -------------------------------------------------------------
    logger.info("Sewing triangles into shell…")

    t0 = perf_counter()
    shell, skipped = make_shell_from_faces(
        vertices=vertices,
        faces_chunk=faces,
        sew_tol=sew_tol,
        faces_area=faces_area
    )
    logger.info(
        f"Sewing completed. Faces skipped due to tiny area: {skipped}"
    )
    logger.debug(
        f"Sewing time: {perf_counter() - t0:.3f} s"
    )

    # -------------------------------------------------------------
    # 2) Extract a closed (watertight) shell
    # -------------------------------------------------------------
    logger.info("Searching for closed shell…")

    t1 = perf_counter()
    closed_shell = None
    exp = TopExp_Explorer(shell, TopAbs_SHELL)

    while exp.More():
        sh = exp.Current()

         if sh.ShapeType() == TopAbs_SHELL: # sanity check: is it a shell?
            if gyroid_utils.occ_tools.is_shell_closed(sh):  # if yes, is it closed?
                closed_shell = sh
                print("Closed shell found.")
                break

        exp.Next()

    if closed_shell is None:
        logger.error("❌ No watertight shell found after sewing.")
        raise RuntimeError("faces_to_solid(): Could not find watertight shell.")

    logger.debug(
        f"Closed-shell extraction time: {perf_counter() - t1:.3f} s"
    )

    # -------------------------------------------------------------
    # 3) Build initial (unsimplified) solid
    # -------------------------------------------------------------
    logger.info("Building initial solid from closed shell…")

    t2 = perf_counter()
    fixer = ShapeFix_Shell()
    fixer.Init(closed_shell)
    fixer.Perform()

    solid_normal = BRepBuilderAPI_MakeSolid(fixer.Shell()).Solid()

    logger.info("Initial solid constructed.")
    logger.debug(
        f"Solid build time: {perf_counter() - t2:.3f} s"
    )

    # -------------------------------------------------------------
    # 4) Simplify shell geometry/topology
    # -------------------------------------------------------------
    logger.info("Simplifying shell…")

    t3 = perf_counter()
    closed_shell = simplify_shell(
        closed_shell,
        linear_tol=sew_tol,
        angular_tol_deg=1
    )

    logger.debug(
        f"Simplification time: {perf_counter() - t3:.3f} s"
    )

    # -------------------------------------------------------------
    # 5) Build simplified solid
    # -------------------------------------------------------------
    logger.info("Building simplified solid…")

    t4 = perf_counter()
    fixer = ShapeFix_Shell()
    fixer.Init(closed_shell)
    fixer.Perform()

    solid_simplified = BRepBuilderAPI_MakeSolid(fixer.Shell()).Solid()

    logger.info("Simplified solid constructed.")
    logger.debug(
        f"Simplified solid build time: {perf_counter() - t4:.3f} s"
    )

    # -------------------------------------------------------------
    # Total runtime
    # -------------------------------------------------------------
    t_total = perf_counter() - t_start
    logger.info(f"faces_to_solid() completed in {t_total:.2f} s")

    return solid_normal, solid_simplified

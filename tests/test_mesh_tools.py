"""
Tests for gyroid_utils.mesh_tools.
"""
import numpy as np
import pytest

#this is just like the import in gyroid_utils.mesh_tools, but we do it here so that pytest can skip all tests in this module if gyroid_utils isn't importable
mesh_tools = pytest.importorskip("gyroid_utils.mesh_tools")

"""
============================================================================
0 - TestCalculateTriangleAreas
1 - TestKeepLargestConnectedComponent
2 - TestSimplifyMesh
3 - TestMeshFromMatrix
4 - TestMatrixFromMesh
============================================================================
"""

# ============================================================================
# 0 - TestCalculateTriangleAreas
# ============================================================================
class TestCalculateTriangleAreas:
    """
    Tests for mesh_tools.calculate_triangle_areas().

    The formula (0.5 * |cross(v1-v0, v2-v0)|) is basic geometry, so these
    check it against triangles with a hand-computable area, plus the
    None/empty-input guards, which raise explicit TypeError/ValueError
    rather than returning some sentinel the caller has to remember to check.
    """

    def test_single_unit_right_triangle(self):
        """A right triangle with legs of length 1 has area 0.5 — the textbook case."""
        verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)
        faces = np.array([[0, 1, 2]])
        areas = mesh_tools.calculate_triangle_areas(verts, faces)
        np.testing.assert_allclose(areas, [0.5])

    def test_multiple_triangles(self):
        """Two right triangles with legs of length 2 (splitting a 2x2 square) should each have area 2.0, computed independently per row."""
        verts = np.array(
            [[0, 0, 0], [2, 0, 0], [0, 2, 0], [2, 2, 0]], dtype=float
        )
        faces = np.array([[0, 1, 2], [1, 3, 2]])
        areas = mesh_tools.calculate_triangle_areas(verts, faces)
        np.testing.assert_allclose(areas, [2.0, 2.0])

    def test_empty_faces_raises_value_error(self):
        """A mesh with zero faces has nothing to compute an area for, so it should raise ValueError rather than divide by zero or return an empty array silently."""
        verts = np.zeros((0, 3))
        faces = np.zeros((0, 3), dtype=int)
        with pytest.raises(ValueError):
            mesh_tools.calculate_triangle_areas(verts, faces)

    def test_none_input_raises_type_error(self):
        """verts=None/faces=None should raise TypeError, per the explicit guard, rather than fail later with a confusing AttributeError."""
        with pytest.raises(TypeError):
            mesh_tools.calculate_triangle_areas(None, None)


# ============================================================================
# 1 - TestKeepLargestConnectedComponent
# ============================================================================
class TestKeepLargestConnectedComponent:
    """
    Tests for mesh_tools.keep_largest_connected_component().

    GyroidModel.simplify_mesh() calls this after decimation to discard stray
    disconnected fragments (a common marching-cubes artifact). These check
    the None/empty guards (which raise explicit TypeError/ValueError rather
    than returning an empty sentinel), and that it actually keeps the
    component with the most faces (that's the "largest" metric the code
    uses — not volume or surface area) rather than, say, whichever
    component appears first.
    """

    def test_none_input_raises_type_error(self):
        """verts=None/faces=None should raise TypeError, per the explicit guard, rather than fail later with a confusing AttributeError."""
        with pytest.raises(TypeError):
            mesh_tools.keep_largest_connected_component(None, None)

    def test_zero_faces_raises_value_error(self):
        """A mesh with vertices but zero faces has nothing to split into components, so it should raise ValueError rather than silently return an empty mesh."""
        verts = np.zeros((3, 3))
        faces = np.zeros((0, 3), dtype=int)
        with pytest.raises(ValueError):
            mesh_tools.keep_largest_connected_component(verts, faces)

    def test_keeps_component_with_more_faces(self):
        """
        "Largest" here means most faces (see keep_largest_connected_component's
        `max(components, key=lambda m: len(m.faces))`), not greatest volume
        or surface area. So this builds two disjoint shapes that actually
        differ in face count: a tetrahedron (4 faces) near the origin, and a
        translated/scaled octahedron (8 faces) far away. They share no
        vertices, so they're guaranteed to split into exactly 2 components;
        only the octahedron's 8 faces should survive.
        """
        tetra_verts = np.array(
            [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float
        )
        tetra_faces = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])

        octa_verts = np.array(
            [
                [1, 0, 0], [-1, 0, 0],
                [0, 1, 0], [0, -1, 0],
                [0, 0, 1], [0, 0, -1],
            ],
            dtype=float,
        ) * 10 + np.array([100.0, 100.0, 100.0])
        octa_faces = np.array(
            [
                [0, 2, 4], [2, 1, 4], [1, 3, 4], [3, 0, 4],
                [2, 0, 5], [1, 2, 5], [3, 1, 5], [0, 3, 5],
            ]
        ) + len(tetra_verts)  # offset to index into the combined vertex array

        verts = np.vstack([tetra_verts, octa_verts])
        faces = np.vstack([tetra_faces, octa_faces])

        v2, f2 = mesh_tools.keep_largest_connected_component(verts, faces)
        assert len(f2) == 8
        assert np.all(v2 >= 90)  # only the translated/large octahedron remains


# ============================================================================
# 2 - TestSimplifyMesh
# ============================================================================
class TestSimplifyMesh:
    """
    Tests for mesh_tools.simplify_mesh().

    simplify_mesh() takes and returns (verts, faces), matching the
    (verts, faces) convention used everywhere else in mesh_tools.py (e.g.
    mesh_from_matrix, keep_largest_connected_component). It used to return
    (faces, verts) instead - the opposite order - which was an easy trap for
    any caller that (reasonably) assumed the same order as the rest of the
    module: verts silently ended up holding face indices and faces silently
    ended up holding vertex coordinates, and the mistake wouldn't surface
    until something far downstream (e.g. viz.save_mesh_as_html) tried to use
    the vertex-coordinate array as an index and crashed with a confusing
    IndexError. test_returns_verts_then_faces_in_that_order locks in the
    correct order so that bug can't silently come back.
    """

    def test_none_input_raises_type_error(self):
        """verts=None/faces=None should raise TypeError, per the explicit guard, rather than fail later with a confusing AttributeError."""
        with pytest.raises(TypeError):
            mesh_tools.simplify_mesh(None, None, target=100)

    def test_zero_faces_raises_type_error(self):
        """A mesh with zero faces has nothing to simplify, so it should raise TypeError rather than silently return an empty mesh."""
        verts = np.zeros((0, 3))
        faces = np.zeros((0, 3), dtype=int)
        with pytest.raises(TypeError):
            mesh_tools.simplify_mesh(verts, faces, target=100)

    def test_invalid_mode_raises_value_error(self):
        """An unrecognized mode string should raise ValueError, not silently fall back to some default simplification method."""
        verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
        faces = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])
        with pytest.raises(ValueError):
            mesh_tools.simplify_mesh(verts, faces, target=100, mode="bogus")

    def test_returns_verts_then_faces_in_that_order(self):
        """
        Regression test for the swapped-return-order bug: with a target face
        count above the mesh's actual face count, "trimesh" mode's decimation
        loop never runs, so the input mesh should come back unchanged. This
        pins down not just that the values are unchanged, but that they come
        back in (verts, faces) order - verts as (N, 3) float coordinates,
        faces as (M, 3) integer indices - rather than swapped.
        """
        verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
        faces = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])

        verts_out, faces_out = mesh_tools.simplify_mesh(verts, faces, target=100, mode="trimesh")

        np.testing.assert_array_equal(verts_out, verts)
        np.testing.assert_array_equal(faces_out, faces)
        assert verts_out.shape[1] == 3
        assert faces_out.shape[1] == 3
        assert np.issubdtype(faces_out.dtype, np.integer)


# ============================================================================
# 3 - TestMeshFromMatrix
# ============================================================================
class TestMeshFromMatrix:
    """
    Tests for mesh_tools.mesh_from_matrix(), the marching-cubes isosurface
    extraction that turns a scalar field into (verts, faces).

    The main test checks the extracted surface is geometrically sane on a
    field whose isosurface is a known shape (a sphere) rather than just
    checking "it returns something." The second test checks the function's
    own error handling: a bad argument should raise a clear RuntimeError
    (wrapping the underlying np.pad failure) rather than a confusing
    lower-level exception.
    """

    def test_extracts_sphere_like_isosurface(self):
        """
        field = 1 - (x^2+y^2+z^2) is positive inside the unit sphere and
        negative outside it, so its zero-isosurface is a unit sphere.
        Extracted vertices should all lie close to radius 1 from the origin.
        """
        n = 20
        lin = np.linspace(-1, 1, n)
        x, y, z = np.meshgrid(lin, lin, lin, indexing="ij")
        field = 1.0 - (x**2 + y**2 + z**2)  # positive inside the unit sphere

        verts, faces = mesh_tools.mesh_from_matrix(
            matrix=field,
            iso_level=0.0,
            algo_step_size=1,
            x=x, y=y, z=z,
            pad_width=2,
        )
        assert verts is not None and faces is not None
        assert len(faces) > 0
        radii = np.linalg.norm(verts, axis=1)
        assert radii.max() < 1.5
        assert radii.min() > 0.3

    def test_invalid_pad_width_raises_runtime_error(self):
        """A negative pad_width makes the internal np.pad() call raise; mesh_from_matrix should catch it and re-raise as a RuntimeError ("Failed to pad matrix."), not propagate np.pad's own ValueError directly."""
        n = 6
        lin = np.linspace(-1, 1, n)
        x, y, z = np.meshgrid(lin, lin, lin, indexing="ij")
        field = np.random.default_rng(0).random((n, n, n))
        with pytest.raises(RuntimeError):
            mesh_tools.mesh_from_matrix(
                matrix=field, iso_level=0.0, algo_step_size=1,
                x=x, y=y, z=z, pad_width=-1,
            )


# ============================================================================
# 4 - TestMatrixFromMesh
# ============================================================================
class TestMatrixFromMesh:
    """
    Tests for mesh_tools.matrix_from_mesh(), the voxelization function that
    inverts mesh_from_matrix()'s marching-cubes extraction. Together the two
    functions form a full loop between the two representations gyroid_utils
    works with: matrix -> mesh (mesh_from_matrix) -> matrix (matrix_from_mesh).
    The main test here exercises that full loop end-to-end - build a
    scalar field, extract its mesh, voxelize the mesh back into a matrix -
    rather than just testing matrix_from_mesh in isolation against a mesh
    built by hand.
    """

    def test_resolution_must_be_positive(self):
        """resolution<=0 has no meaningful voxel pitch, so it should raise ValueError rather than divide by zero or produce a nonsensical grid."""
        verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
        faces = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])
        with pytest.raises(ValueError):
            mesh_tools.matrix_from_mesh(verts, faces, resolution=0)

    def test_zero_span_mesh_raises_value_error(self):
        """A mesh flattened onto a single plane (zero extent along z) can't be voxelized into a 3D grid, so it should raise ValueError rather than produce a degenerate matrix."""
        verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=float)
        faces = np.array([[0, 1, 2], [1, 3, 2]])
        with pytest.raises(ValueError):
            mesh_tools.matrix_from_mesh(verts, faces, resolution=16)

    def test_matrix_mesh_matrix_round_trip_preserves_sphere(self):
        """
        Full loop: build a scalar field whose isosurface is a unit sphere
        (same field as TestMeshFromMatrix.test_extracts_sphere_like_isosurface),
        extract its mesh with mesh_from_matrix(), then voxelize that mesh back
        into a matrix with matrix_from_mesh(). The occupied voxels of the
        round-tripped matrix should still describe (roughly) the same unit
        sphere: nothing occupied far outside radius 1, and the interior
        (near-zero radius) should actually be filled in, not just the shell,
        since matrix_from_mesh().fill()'s whole point is a solid volume.
        """
        n = 20
        lin = np.linspace(-1, 1, n)
        x, y, z = np.meshgrid(lin, lin, lin, indexing="ij")
        field = 1.0 - (x**2 + y**2 + z**2)  # positive inside the unit sphere

        verts, faces = mesh_tools.mesh_from_matrix(
            matrix=field,
            iso_level=0.0,
            algo_step_size=1,
            x=x, y=y, z=z,
            pad_width=2,
        )

        vx, vy, vz, matrix = mesh_tools.matrix_from_mesh(verts, faces, resolution=20)

        assert matrix.shape == (len(vx), len(vy), len(vz))
        assert matrix.any()  # something got voxelized and filled in

        gx, gy, gz = np.meshgrid(vx, vy, vz, indexing="ij")
        occupied_radii = np.sqrt(gx[matrix] ** 2 + gy[matrix] ** 2 + gz[matrix] ** 2)

        assert occupied_radii.max() < 1.5  # nothing filled well outside the sphere
        assert occupied_radii.min() < 0.3  # the center is filled, not just the shell

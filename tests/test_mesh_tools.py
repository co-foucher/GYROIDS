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
2 - TestMeshFromMatrix
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
# 2 - TestMeshFromMatrix
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
            pad_val=-1.0,
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

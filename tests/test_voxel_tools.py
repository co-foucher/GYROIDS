"""
Tests for gyroid_utils.voxel_tools.
"""
import numpy as np
import pytest

#this is just like the import in gyroid_utils.voxel_tools, but we do it here so that pytest can skip all tests in this module if gyroid_utils isn't importable
voxel_tools = pytest.importorskip("gyroid_utils.voxel_tools")

"""
============================================================================
0 - TestInterpolateVoxelGrid
============================================================================
"""

# ============================================================================
# 0 - TestInterpolateVoxelGrid
# ============================================================================
class TestInterpolateVoxelGrid:
    """
    Tests for voxel_tools.interpolate_voxel_grid().

    interpolate_voxel_grid() resamples a 3D array onto a new (x_dim, y_dim,
    z_dim) grid via scipy.ndimage.zoom with trilinear interpolation
    (order=1). These checks cover: the output shape landing exactly on the
    requested dimensions, both up- and down-sampling; that a linear field is
    reproduced exactly (linear interpolation of linear data is exact -- the
    textbook case, and different sizes per axis also catch an axis-order
    swap in zoom_factors); that resampled values never exceed the input's
    min/max (a basic convexity property of linear interpolation); that
    interpolating a binary voxel grid produces intermediate fractional
    values at solid/empty boundaries (the behavior called out in the
    function's NOTE, not a nearest-neighbor resample); and the ndim guard.
    """

    def test_upsample_shape(self):
        """Upsampling a (4,5,6) grid to (8,10,12) should return exactly that shape."""
        grid = np.zeros((4, 5, 6))
        out = voxel_tools.interpolate_voxel_grid(grid, 8, 10, 12)
        assert out.shape == (8, 10, 12)

    def test_downsample_shape(self):
        """Downsampling a (8,10,12) grid to (4,5,6) should return exactly that shape."""
        grid = np.zeros((8, 10, 12))
        out = voxel_tools.interpolate_voxel_grid(grid, 4, 5, 6)
        assert out.shape == (4, 5, 6)

    def test_linear_field_reproduced_exactly(self):
        """
        A field that is exactly linear in each axis (field[i,j,k] = i + 2*j + 3*k)
        is the textbook case for trilinear interpolation: linear interpolation
        of a linear function is exact everywhere within the sampled range, so
        the resampled grid should match the same formula evaluated at the new
        (fractional) voxel indices, up to floating-point error. Using a
        different target size per axis also catches an axis-order swap bug
        (e.g. x/y/z mixed up when building zoom_factors).
        """
        nx, ny, nz = 4, 5, 6
        grid = np.fromfunction(lambda i, j, k: i + 2 * j + 3 * k, (nx, ny, nz))
        tx, ty, tz = 8, 10, 12
        out = voxel_tools.interpolate_voxel_grid(grid, tx, ty, tz)

        ix = np.linspace(0, nx - 1, tx)
        iy = np.linspace(0, ny - 1, ty)
        iz = np.linspace(0, nz - 1, tz)
        I, J, K = np.meshgrid(ix, iy, iz, indexing="ij")
        expected = I + 2 * J + 3 * K
        np.testing.assert_allclose(out, expected, atol=1e-10)

    def test_output_stays_within_input_range(self):
        """Trilinear interpolation only takes convex combinations of neighboring input values, so no resampled value -- up- or down-sampled -- should fall outside [min(input), max(input)]."""
        rng = np.random.default_rng(0)
        grid = rng.uniform(-3.0, 5.0, size=(6, 7, 8))
        out_up = voxel_tools.interpolate_voxel_grid(grid, 12, 14, 16)
        out_down = voxel_tools.interpolate_voxel_grid(grid, 3, 4, 5)
        for out in (out_up, out_down):
            assert out.min() >= grid.min() - 1e-10
            assert out.max() <= grid.max() + 1e-10

    def test_binary_voxel_grid_gets_fractional_boundary_values(self):
        """
        Per the function's NOTE, interpolating a strictly binary (0/1) voxel
        grid does NOT stay binary: with order=1 the solid/empty boundary is
        blended, producing values strictly between 0 and 1. This pins down
        that documented behavior so a future switch back to nearest-neighbor
        doesn't silently break callers relying on the blend.
        """
        grid = np.zeros((8, 3, 3))
        grid[4:, :, :] = 1.0  # empty for the first half along x, solid for the second

        out = voxel_tools.interpolate_voxel_grid(grid, 16, 3, 3)
        profile = out[:, 0, 0]

        assert profile[0] == 0.0, "far from the boundary, should still be exactly empty"
        assert profile[-1] == 1.0, "far from the boundary, should still be exactly solid"
        assert np.any((profile > 0.0) & (profile < 1.0)), "expected fractional values at the solid/empty boundary"

    def test_non_3d_input_raises_value_error(self):
        """A 2D or 4D array isn't a voxel/field grid, so this should raise ValueError rather than fail deeper inside scipy.ndimage.zoom with a confusing error."""
        with pytest.raises(ValueError):
            voxel_tools.interpolate_voxel_grid(np.zeros((4, 4)), 8, 8, 8)
        with pytest.raises(ValueError):
            voxel_tools.interpolate_voxel_grid(np.zeros((4, 4, 4, 4)), 8, 8, 8)

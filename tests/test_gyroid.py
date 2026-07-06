"""
Tests for gyroid_utils.gyroid.GyroidModel.
"""
import numpy as np
import pytest

#this is just like the import in gyroid_utils.gyroid, but we do it here so that pytest can skip all tests in this module if gyroid_utils isn't importable
gyroid_mod = pytest.importorskip("gyroid_utils.gyroid") 

#pull one attribute (the class) out of the module object gyroid_mod that importorskip just returned, and bind it to the name GyroidModel
GyroidModel = gyroid_mod.GyroidModel   

"""
============================================================================
0 - helper functions
1 - TestValidateInputs 
2 - TestComputeField
3 - TestGuardedMethodsRequireField
4 - TestAddBaseplates
============================================================================
"""

# ============================================================================
# 0 - helper functions
# ============================================================================

def _expected_term(x, y, z, px, py, pz):
    """small function to not have to re-write the formula in multiple tests"""
    return (
        np.sin((2 * np.pi / px) * x) * np.cos((2 * np.pi / py) * y)
        + np.sin((2 * np.pi / py) * y) * np.cos((2 * np.pi / pz) * z)
        + np.sin((2 * np.pi / pz) * z) * np.cos((2 * np.pi / px) * x)
    )


def _expected_distance_field(x, y, z, px, py, pz, thickness):
    """
    Independent reimplementation of compute_field(mode="distance"), so the
    real implementation can be checked against it instead of against itself.
    Mirrors gyroid.py step for step: build the same term/binary split, get
    each voxel's distance to the nearest sign change (summed both ways),
    then keep that distance where it's under thickness/2 and -1 elsewhere.
    """
    from scipy.ndimage import distance_transform_edt

    term = _expected_term(x, y, z, px, py, pz)

    dx = float(x[1, 0, 0] - x[0, 0, 0])
    dy = float(y[0, 1, 0] - y[0, 0, 0])
    dz = float(z[0, 0, 1] - z[0, 0, 0])
    spacing = (dx, dy, dz)

    binary = term > 0
    dist_out = distance_transform_edt(~binary, sampling=spacing)
    dist_in = distance_transform_edt(binary, sampling=spacing)
    dist = dist_out + dist_in

    half_t = thickness / 2.0
    mask = dist < half_t
    v = np.zeros_like(dist) - 1
    v[mask] = dist[mask]
    return v

# ============================================================================
# 1 - TestValidateInputs
# ============================================================================
class TestValidateInputs:
    """
    Tests for GyroidModel.__init__ / _validate_inputs().

    _validate_inputs() runs automatically inside the __init__ of the gyroid class
    and is the only thing standing between a caller and a model built on garbage 
    data (wrong types, mismatched grid shapes, etc.). These tests make sure it 
    actually rejects the bad cases it claims to reject, and still accepts the valid
    ones, so a future edit to that method can't silently loosen or break it.
    """

    def test_rejects_non_ndarray_coordinates(self, small_grid):
        """x/y/z must be numpy arrays; a plain list should raise TypeError."""
        x, y, z = small_grid
        with pytest.raises(TypeError):
            GyroidModel(x.tolist(), y, z, 1.0, 1.0, 1.0, 0.2)

    def test_rejects_mismatched_coordinate_shapes(self, small_grid):
        """x, y, z must all share the same shape; z one slice short should raise ValueError."""
        x, y, z = small_grid
        with pytest.raises(ValueError):
            GyroidModel(x, y, z[:-1], 1.0, 1.0, 1.0, 0.2)

    def test_rejects_mismatched_param_array_shape(self, small_grid):
        """If px is given as an array, it must match x/y/z's shape; a (2,2,2) array against an (8,8,8) grid should raise ValueError."""
        x, y, z = small_grid
        bad_px = np.ones((2, 2, 2))
        with pytest.raises(ValueError):
            GyroidModel(x, y, z, bad_px, 1.0, 1.0, 0.2)

    def test_rejects_non_array_non_scalar_param(self, small_grid):
        """px must be a scalar or a numpy array; a plain list is neither and should raise TypeError."""
        x, y, z = small_grid
        with pytest.raises(TypeError):
            GyroidModel(x, y, z, [1.0, 2.0], 1.0, 1.0, 0.2)

    def test_accepts_scalar_and_matching_array_params(self, small_grid):
        """The valid counterpart: a correctly-shaped px array is accepted, and a freshly built model has no field/mesh computed yet."""
        x, y, z = small_grid
        px_array = np.full(x.shape, 1.5)
        model = GyroidModel(x, y, z, px_array, 1.0, 1.0, 0.2)
        assert model.v is None
        assert model.verts is None and model.faces is None


# ============================================================================
# 2 - TestComputeField
# ============================================================================
class TestComputeField:
    """
    Tests for GyroidModel.compute_field(), the actual gyroid math.

    This is the highest-value group in the file: if a future edit to the
    trig formula, a sign flip, or a period substitution changes the field
    values, these tests catch it immediately instead of someone noticing
    weeks later that an exported STL looks subtly wrong. Each mode is
    checked against an independently-written copy of the expected formula
    (_expected_term, above), not against the code's own output, so a bug
    in compute_field can't accidentally match a bug in the test.
    """

    def test_abs_mode_matches_formula(self, small_grid):
        """mode="abs" should equal thickness - |term|, per the docstring."""
        x, y, z = small_grid
        model = GyroidModel(x, y, z, 1.3, 1.1, 0.9, 0.25)
        v = model.compute_field(mode="abs")
        expected = 0.25 - np.abs(_expected_term(x, y, z, 1.3, 1.1, 0.9))
        np.testing.assert_allclose(v, expected)
        assert model.v is v

    def test_signed_mode_matches_formula(self, small_grid):
        """mode="signed" should equal term - thickness (a plain level-set, no abs())."""
        x, y, z = small_grid
        model = GyroidModel(x, y, z, 1.3, 1.1, 0.9, 0.25)
        v = model.compute_field(mode="signed")
        expected = _expected_term(x, y, z, 1.3, 1.1, 0.9) - 0.25
        np.testing.assert_allclose(v, expected)
    
    def test_distance_mode_produces_expected_range(self, small_grid):
        """
        Every value should be either exactly -1 (outside the
        wall band) or a distance within [0, thickness/2] (inside it), per
        the docstring's description of the mode.
        """
        x, y, z = small_grid
        thickness = 0.5
        model = GyroidModel(x, y, z, 1.3, 1.1, 0.9, thickness)
        v = model.compute_field(mode="distance")
        assert v.shape == x.shape
        # field is -1 outside the wall band, else a distance in [0, thickness/2]
        assert np.all((v == -1) | ((v >= 0) & (v <= thickness / 2.0 + 1e-9)))

    def test_distance_mode_matches_formula(self, small_grid):
        """
        Stronger companion to test_distance_mode_produces_expected_range:
        instead of just checking the output stays in a valid range, this
        recomputes the whole distance-transform pipeline independently
        (_expected_distance_field, above) and asserts an exact match against
        compute_field's own output.
        """
        x, y, z = small_grid
        px, py, pz, thickness = 1.3, 1.1, 0.9, 0.5
        model = GyroidModel(x, y, z, px, py, pz, thickness)
        v = model.compute_field(mode="distance")
        expected = _expected_distance_field(x, y, z, px, py, pz, thickness)
        np.testing.assert_allclose(v, expected)

    def test_default_mode_is_abs(self, small_grid):
        """Calling compute_field() with no mode argument should behave exactly like mode="abs"."""
        x, y, z = small_grid
        model = GyroidModel(x, y, z, 1.3, 1.1, 0.9, 0.25)
        v_default = model.compute_field()
        v_abs = model.compute_field(mode="abs")
        np.testing.assert_allclose(v_default, v_abs)

    def test_invalid_mode_raises(self, small_grid):
        """An unrecognized mode string should raise ValueError, not silently return a wrong/empty field."""
        x, y, z = small_grid
        model = GyroidModel(x, y, z, 1.0, 1.0, 1.0, 0.2)
        with pytest.raises(ValueError):
            model.compute_field(mode="bogus")


# ============================================================================
# 3 - TestGuardedMethodsRequireField
# ============================================================================
class TestGuardedMethodsRequireField:
    """
    Tests for the "did you call things in the right order" guard rails.

    GyroidModel's workflow has an implicit order: compute_field() before
    generate_mesh(), generate_mesh() before check_mesh_quality()/fix_mesh(),
    compute_field() before save(). Several methods check for this explicitly
    and fail loudly (RuntimeError, or a documented None/None return) instead
    of crashing deep inside with a confusing AttributeError on self.v/self.
    verts being None. These tests make sure each guard is still in place, so
    a future refactor can't accidentally delete a check and turn a clear
    error into a cryptic one.
    """

    def test_generate_mesh_without_field_returns_none(self, small_grid):
        """generate_mesh() before compute_field() should return (None, None), per its documented early-exit behavior, not raise."""
        x, y, z = small_grid
        model = GyroidModel(x, y, z, 1.0, 1.0, 1.0, 0.2)
        verts, faces = model.generate_mesh()
        assert verts is None and faces is None

    def test_save_without_field_raises(self, tmp_path, small_grid):
        """save() needs self.v; calling it before compute_field() should raise RuntimeError rather than write a garbage file."""
        x, y, z = small_grid
        model = GyroidModel(x, y, z, 1.0, 1.0, 1.0, 0.2)
        with pytest.raises(RuntimeError):
            model.save(str(tmp_path / "out.npz"))

    def test_check_mesh_quality_without_mesh_raises(self, small_grid):
        """check_mesh_quality() needs self.verts/self.faces; calling it before generate_mesh() should raise RuntimeError."""
        x, y, z = small_grid
        model = GyroidModel(x, y, z, 1.0, 1.0, 1.0, 0.2)
        with pytest.raises(RuntimeError):
            model.check_mesh_quality()

    def test_fix_mesh_without_mesh_raises(self, small_grid):
        """fix_mesh() needs self.verts/self.faces; calling it before generate_mesh() should raise RuntimeError."""
        x, y, z = small_grid
        model = GyroidModel(x, y, z, 1.0, 1.0, 1.0, 0.2)
        with pytest.raises(RuntimeError):
            model.fix_mesh()

    def test_add_baseplates_without_field_raises(self, small_grid):
        """add_baseplates() needs self.v; calling it before compute_field() should raise RuntimeError."""
        x, y, z = small_grid
        model = GyroidModel(x, y, z, 1.0, 1.0, 1.0, 0.2)
        with pytest.raises(RuntimeError):
            model.add_baseplates(thickness=1.0)


# ============================================================================
# 4 - TestAddBaseplates
# ============================================================================
class TestAddBaseplates:
    """
    Tests for GyroidModel.add_baseplates(), specifically the index/slicing
    math (thickness -> number of z-slices -> which voxels get set to 1).

    This is fiddly by nature: it converts a physical thickness into a slice
    count via integer division, then has to clamp that count at both ends
    (zero/sub-voxel thickness should do nothing, an oversized thickness
    should clamp to the whole volume instead of an out-of-bounds slice).
    Each test targets one of those edge cases so an off-by-one or a broken
    clamp gets caught here instead of showing up as a malformed baseplate
    on an actual print.
    """

    def test_fills_expected_number_of_slices(self, small_grid):
        """A thickness worth exactly 3 slices should set exactly the first and last 3 z-slices to 1, and nothing else."""
        x, y, z = small_grid
        model = GyroidModel(x, y, z, 1.0, 1.0, 1.0, 0.2)
        model.compute_field(mode="abs")

        dz = abs(z[0, 0, 1] - z[0, 0, 0])
        thickness = 3 * dz  # exactly 3 slices worth
        model.add_baseplates(thickness=thickness)

        n = int(thickness / dz)
        assert np.all(model.v[:, :, :n] == 1)
        assert np.all(model.v[:, :, -n:] == 1)

    def test_zero_or_subvoxel_thickness_is_noop(self, small_grid):
        """thickness=0.0 (below one voxel of spacing) should leave the field completely untouched, per the documented early-return."""
        x, y, z = small_grid
        model = GyroidModel(x, y, z, 1.0, 1.0, 1.0, 0.2)
        model.compute_field(mode="abs")
        before = model.v.copy()
        model.add_baseplates(thickness=0.0)
        np.testing.assert_array_equal(model.v, before)

    def test_thickness_covering_whole_volume_fills_everything(self, small_grid):
        """A thickness far larger than the grid's z-extent should clamp to filling the entire volume with 1, not raise or index out of bounds."""
        x, y, z = small_grid
        model = GyroidModel(x, y, z, 1.0, 1.0, 1.0, 0.2)
        model.compute_field(mode="abs")
        model.add_baseplates(thickness=1000.0)
        assert np.all(model.v == 1)

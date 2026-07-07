"""
Tests for gyroid_utils.tpms_base.TPMSModel.

TPMSModel is the shared base class behind every concrete TPMS type
(GyroidModel, SchwartzPModel, DiamondModel, IWPModel, ...). Validation, the
abs/signed/distance field pipeline, guarded methods, and add_baseplates all
live here now instead of being duplicated per subclass, so they're tested
once, through a minimal concrete subclass built just for these tests
(_DummyTPMS) rather than through any real surface. Each real surface's own
implicit-equation formula is tested separately in test_tpms_surfaces.py.
"""
import numpy as np
import pytest

#this is just like the import in gyroid_utils.tpms_base, but we do it here so that pytest can skip all tests in this module if gyroid_utils isn't importable
tpms_base = pytest.importorskip("gyroid_utils.tpms_base")
TPMSModel = tpms_base.TPMSModel

"""
============================================================================
0 - _DummyTPMS + helper functions
1 - TestSurfaceTermIsAbstract
2 - TestValidateInputs
3 - TestComputeField
4 - TestGuardedMethodsRequireField
5 - TestAddBaseplates
============================================================================
"""

# ============================================================================
# 0 - _DummyTPMS + helper functions
# ============================================================================

class _DummyTPMS(TPMSModel):
    """
    Minimal concrete TPMSModel, used only to exercise the base class's
    shared logic in isolation from any real surface's formula. This is not
    a real TPMS type - the "surface" here is just some periodic trig
    combination, picked to be simple to hand-verify and to oscillate in
    sign (needed to exercise the abs/signed/distance modes meaningfully).
    """
    DEFAULT_FIELD_MODE = "abs"

    def _surface_term(self):
        return (
            np.sin((2 * np.pi / self.px) * self.x)
            + np.cos((2 * np.pi / self.py) * self.y)
            + np.sin((2 * np.pi / self.pz) * self.z)
        )


def _dummy_term(x, y, z, px, py, pz):
    """Independent reimplementation of _DummyTPMS._surface_term, for formula checks."""
    return (
        np.sin((2 * np.pi / px) * x)
        + np.cos((2 * np.pi / py) * y)
        + np.sin((2 * np.pi / pz) * z)
    )


def _expected_distance_field(term, x, y, z, thickness):
    """
    Independent reimplementation of TPMSModel.compute_field(mode="distance"),
    given an already-computed term array, so the real implementation can be
    checked against it instead of against itself.
    """
    from scipy.ndimage import distance_transform_edt

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
# 1 - TestSurfaceTermIsAbstract
# ============================================================================
class TestSurfaceTermIsAbstract:
    """_surface_term() must be overridden by a subclass; the base class's own version should refuse to run."""

    def test_raw_tpmsmodel_raises_not_implemented(self, small_grid):
        """Instantiating TPMSModel directly (no subclass) is fine, but calling compute_field() must raise NotImplementedError since _surface_term() was never overridden."""
        x, y, z = small_grid
        model = TPMSModel(x, y, z, 1.0, 1.0, 1.0, 0.2)
        with pytest.raises(NotImplementedError):
            model.compute_field()


# ============================================================================
# 2 - TestValidateInputs
# ============================================================================
class TestValidateInputs:
    """
    Tests for TPMSModel.__init__ / _validate_inputs(), shared by every TPMS
    subclass. Exercised through _DummyTPMS since this validation logic
    doesn't depend on which surface formula is used.
    """

    def test_rejects_non_ndarray_coordinates(self, small_grid):
        """x/y/z must be numpy arrays; a plain list should raise TypeError."""
        x, y, z = small_grid
        with pytest.raises(TypeError):
            _DummyTPMS(x.tolist(), y, z, 1.0, 1.0, 1.0, 0.2)

    def test_rejects_mismatched_coordinate_shapes(self, small_grid):
        """x, y, z must all share the same shape; z one slice short should raise ValueError."""
        x, y, z = small_grid
        with pytest.raises(ValueError):
            _DummyTPMS(x, y, z[:-1], 1.0, 1.0, 1.0, 0.2)

    def test_rejects_mismatched_param_array_shape(self, small_grid):
        """If px is given as an array, it must match x/y/z's shape; a (2,2,2) array against an (8,8,8) grid should raise ValueError."""
        x, y, z = small_grid
        bad_px = np.ones((2, 2, 2))
        with pytest.raises(ValueError):
            _DummyTPMS(x, y, z, bad_px, 1.0, 1.0, 0.2)

    def test_rejects_non_array_non_scalar_param(self, small_grid):
        """px must be a scalar or a numpy array; a plain list is neither and should raise TypeError."""
        x, y, z = small_grid
        with pytest.raises(TypeError):
            _DummyTPMS(x, y, z, [1.0, 2.0], 1.0, 1.0, 0.2)

    def test_accepts_scalar_and_matching_array_params(self, small_grid):
        """The valid counterpart: a correctly-shaped px array is accepted, and a freshly built model has no field/mesh computed yet."""
        x, y, z = small_grid
        px_array = np.full(x.shape, 1.5)
        model = _DummyTPMS(x, y, z, px_array, 1.0, 1.0, 0.2)
        assert model.v is None
        assert model.verts is None and model.faces is None


# ============================================================================
# 3 - TestComputeField
# ============================================================================
class TestComputeField:
    """
    Tests for TPMSModel.compute_field(): mode dispatch (abs/signed/
    distance/distance_fast/invalid) and the "mode=None -> DEFAULT_FIELD_MODE"
    default lookup. Checked through _DummyTPMS's own simple formula; each
    real surface's formula is separately verified in test_tpms_surfaces.py.
    """

    def test_abs_mode_matches_formula(self, small_grid):
        """mode="abs" should equal thickness - |term|, per the docstring."""
        x, y, z = small_grid
        model = _DummyTPMS(x, y, z, 1.3, 1.1, 0.9, 0.25)
        v = model.compute_field(mode="abs")
        expected = 0.25 - np.abs(_dummy_term(x, y, z, 1.3, 1.1, 0.9))
        np.testing.assert_allclose(v, expected)
        assert model.v is v

    def test_signed_mode_matches_formula(self, small_grid):
        """mode="signed" should equal term - thickness (a plain level-set, no abs())."""
        x, y, z = small_grid
        model = _DummyTPMS(x, y, z, 1.3, 1.1, 0.9, 0.25)
        v = model.compute_field(mode="signed")
        expected = _dummy_term(x, y, z, 1.3, 1.1, 0.9) - 0.25
        np.testing.assert_allclose(v, expected)

    def test_default_mode_uses_default_field_mode_attribute(self, small_grid):
        """Calling compute_field() with no mode argument should use the subclass's DEFAULT_FIELD_MODE ("abs" for _DummyTPMS)."""
        x, y, z = small_grid
        model = _DummyTPMS(x, y, z, 1.3, 1.1, 0.9, 0.25)
        v_default = model.compute_field()
        v_abs = model.compute_field(mode="abs")
        np.testing.assert_allclose(v_default, v_abs)

    def test_invalid_mode_raises(self, small_grid):
        """An unrecognized mode string should raise ValueError, not silently return a wrong/empty field."""
        x, y, z = small_grid
        model = _DummyTPMS(x, y, z, 1.0, 1.0, 1.0, 0.2)
        with pytest.raises(ValueError):
            model.compute_field(mode="bogus")

    def test_distance_mode_matches_formula(self, small_grid):
        """mode="distance" is checked by independently rebuilding the whole distance-transform pipeline (binary split -> distance_transform_edt -> threshold at thickness/2) and asserting an exact match."""
        pytest.importorskip("scipy")
        x, y, z = small_grid
        px, py, pz, thickness = 1.3, 1.1, 0.9, 0.5
        model = _DummyTPMS(x, y, z, px, py, pz, thickness)
        v = model.compute_field(mode="distance")
        term = _dummy_term(x, y, z, px, py, pz)
        expected = _expected_distance_field(term, x, y, z, thickness)
        np.testing.assert_allclose(v, expected)

    def test_distance_fast_mode_produces_expected_range(self, small_grid):
        """mode="distance_fast" (taxicab approximation) should still produce values that are either exactly -1 (outside the wall band) or a distance within [0, thickness/2] (inside it)."""
        x, y, z = small_grid
        thickness = 0.5
        model = _DummyTPMS(x, y, z, 1.3, 1.1, 0.9, thickness)
        v = model.compute_field(mode="distance_fast")
        assert v.shape == x.shape
        assert np.all((v == -1) | ((v >= 0) & (v <= thickness / 2.0 + 1e-9)))


# ============================================================================
# 4 - TestGuardedMethodsRequireField
# ============================================================================
class TestGuardedMethodsRequireField:
    """
    Tests for the "did you call things in the right order" guard rails,
    shared by every TPMS subclass via TPMSModel.
    """

    def test_generate_mesh_without_field_returns_none(self, small_grid):
        """generate_mesh() before compute_field() should return (None, None), per its documented early-exit behavior, not raise."""
        x, y, z = small_grid
        model = _DummyTPMS(x, y, z, 1.0, 1.0, 1.0, 0.2)
        verts, faces = model.generate_mesh()
        assert verts is None and faces is None

    def test_save_without_field_raises(self, tmp_path, small_grid):
        """save() needs self.v; calling it before compute_field() should raise RuntimeError rather than write a garbage file."""
        x, y, z = small_grid
        model = _DummyTPMS(x, y, z, 1.0, 1.0, 1.0, 0.2)
        with pytest.raises(RuntimeError):
            model.save(str(tmp_path / "out.npz"))

    def test_check_mesh_quality_without_mesh_raises(self, small_grid):
        """check_mesh_quality() needs self.verts/self.faces; calling it before generate_mesh() should raise RuntimeError."""
        x, y, z = small_grid
        model = _DummyTPMS(x, y, z, 1.0, 1.0, 1.0, 0.2)
        with pytest.raises(RuntimeError):
            model.check_mesh_quality()

    def test_fix_mesh_without_mesh_raises(self, small_grid):
        """fix_mesh() needs self.verts/self.faces; calling it before generate_mesh() should raise RuntimeError."""
        x, y, z = small_grid
        model = _DummyTPMS(x, y, z, 1.0, 1.0, 1.0, 0.2)
        with pytest.raises(RuntimeError):
            model.fix_mesh()

    def test_add_baseplates_without_field_raises(self, small_grid):
        """add_baseplates() needs self.v; calling it before compute_field() should raise RuntimeError."""
        x, y, z = small_grid
        model = _DummyTPMS(x, y, z, 1.0, 1.0, 1.0, 0.2)
        with pytest.raises(RuntimeError):
            model.add_baseplates(thickness=1.0)


# ============================================================================
# 5 - TestAddBaseplates
# ============================================================================
class TestAddBaseplates:
    """
    Tests for TPMSModel.add_baseplates()'s slicing math (thickness ->
    number of z-slices -> which voxels get set to 1), shared by every TPMS
    subclass.
    """

    def test_fills_expected_number_of_slices(self, small_grid):
        """A thickness worth exactly 3 slices should set exactly the first and last 3 z-slices to 1, and nothing else."""
        x, y, z = small_grid
        model = _DummyTPMS(x, y, z, 1.0, 1.0, 1.0, 0.2)
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
        model = _DummyTPMS(x, y, z, 1.0, 1.0, 1.0, 0.2)
        model.compute_field(mode="abs")
        before = model.v.copy()
        model.add_baseplates(thickness=0.0)
        np.testing.assert_array_equal(model.v, before)

    def test_thickness_covering_whole_volume_fills_everything(self, small_grid):
        """A thickness far larger than the grid's z-extent should clamp to filling the entire volume with 1, not raise or index out of bounds."""
        x, y, z = small_grid
        model = _DummyTPMS(x, y, z, 1.0, 1.0, 1.0, 0.2)
        model.compute_field(mode="abs")
        model.add_baseplates(thickness=1000.0)
        assert np.all(model.v == 1)

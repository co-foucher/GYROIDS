"""
Tests for gyroid_utils.tpms_schwartzp.SchwartzPModel.
"""
import numpy as np
import pytest

schwartzp_mod = pytest.importorskip("gyroid_utils.tpms_schwartzp")
SchwartzPModel = schwartzp_mod.SchwartzPModel


def _expected_term(x, y, z, px, py, pz):
    return (
        np.cos((2 * np.pi / px) * x)
        + np.cos((2 * np.pi / py) * y)
        + np.cos((2 * np.pi / pz) * z)
    )


class TestValidateInputs:
    def test_rejects_non_ndarray_coordinates(self, small_grid):
        x, y, z = small_grid
        with pytest.raises(TypeError):
            SchwartzPModel(x.tolist(), y, z, 1.0, 1.0, 1.0, 0.2)

    def test_rejects_mismatched_coordinate_shapes(self, small_grid):
        x, y, z = small_grid
        with pytest.raises(ValueError):
            SchwartzPModel(x, y, z[:-1], 1.0, 1.0, 1.0, 0.2)

    def test_rejects_mismatched_param_array_shape(self, small_grid):
        x, y, z = small_grid
        bad_thickness = np.ones((2, 2, 2))
        with pytest.raises(ValueError):
            SchwartzPModel(x, y, z, 1.0, 1.0, 1.0, bad_thickness)


class TestComputeField:
    def test_abs_mode_matches_formula(self, small_grid):
        x, y, z = small_grid
        model = SchwartzPModel(x, y, z, 1.3, 1.1, 0.9, 0.25)
        v = model.compute_field(mode="abs")
        expected = 0.25 - np.abs(_expected_term(x, y, z, 1.3, 1.1, 0.9))
        np.testing.assert_allclose(v, expected)

    def test_signed_mode_matches_formula(self, small_grid):
        x, y, z = small_grid
        model = SchwartzPModel(x, y, z, 1.3, 1.1, 0.9, 0.25)
        v = model.compute_field(mode="signed")
        expected = _expected_term(x, y, z, 1.3, 1.1, 0.9) - 0.25
        np.testing.assert_allclose(v, expected)

    def test_field_amplitude_bounds(self, small_grid):
        # F = cos + cos + cos has amplitude range [-3, 3] regardless of grid/periods.
        x, y, z = small_grid
        model = SchwartzPModel(x, y, z, 1.3, 1.1, 0.9, 0.0)
        v = model.compute_field(mode="signed")
        assert v.min() >= -3.0 - 1e-9
        assert v.max() <= 3.0 + 1e-9

    def test_invalid_mode_raises(self, small_grid):
        x, y, z = small_grid
        model = SchwartzPModel(x, y, z, 1.0, 1.0, 1.0, 0.2)
        with pytest.raises(ValueError):
            model.compute_field(mode="bogus")


class TestGuardedMethodsRequireField:
    def test_generate_mesh_without_field_returns_none(self, small_grid):
        x, y, z = small_grid
        model = SchwartzPModel(x, y, z, 1.0, 1.0, 1.0, 0.2)
        verts, faces = model.generate_mesh()
        assert verts is None and faces is None

    def test_add_baseplates_without_field_raises(self, small_grid):
        x, y, z = small_grid
        model = SchwartzPModel(x, y, z, 1.0, 1.0, 1.0, 0.2)
        with pytest.raises(RuntimeError):
            model.add_baseplates(thickness=1.0)

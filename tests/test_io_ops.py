"""
Tests for gyroid_utils.io_ops.
"""
import numpy as np
import pytest

#this is just like the import in gyroid_utils.io_ops, but we do it here so that pytest can skip all tests in this module if gyroid_utils isn't importable
io_ops = pytest.importorskip("gyroid_utils.io_ops")

"""
============================================================================
0 - helper functions
1 - TestGyroidMatricesRoundtrip : load and save of matrices must match
2 - TestGyroidModelSaveLoadRoundtrip
============================================================================
"""

# ============================================================================
# 0 - helper functions
# ============================================================================

def _make_arrays(shape=(4, 4, 4)):
    """small function to not have to re-write the same set of matching-shape arrays in every test"""
    rng = np.random.default_rng(0)
    return {
        "Xres": rng.random(shape),
        "Yres": rng.random(shape),
        "Zres": rng.random(shape),
        "Xperiod": np.full(shape, 1.0),
        "Yperiod": np.full(shape, 1.2),
        "Zperiod": np.full(shape, 0.8),
        "thickness": np.full(shape, 0.3),
        "gyroid_field": rng.random(shape),
    }


# ============================================================================
# 1 - TestGyroidMatricesRoundtrip
# ============================================================================
class TestGyroidMatricesRoundtrip:
    """
    Tests for io_ops.save_gyroid_matrices() / load_gyroid_matrices() directly
    (the low-level .npz read/write, independent of GyroidModel).

    These check the round-trip contract the functions exist for (what you
    save is exactly what you get back), plus the failure paths: a
    shape-mismatched save should raise ValueError without writing a file,
    and a missing/corrupted load should raise a clear, specific exception
    (FileNotFoundError / RuntimeError) rather than fail silently or return
    something a caller could forget to check.
    """

    def test_save_and_load_roundtrip(self, tmp_path):
        """Saving arrays to .npz and loading them back should reproduce the exact same values for every array, in the same order."""
        arrays = _make_arrays()
        outfile = tmp_path / "gyroid_data.npz"
        io_ops.save_gyroid_matrices(str(outfile), **arrays)
        assert outfile.exists()

        loaded = io_ops.load_gyroid_matrices(str(outfile))
        names = [
            "Xres", "Yres", "Zres",
            "Xperiod", "Yperiod", "Zperiod",
            "thickness", "gyroid_field",
        ]
        for name, loaded_arr in zip(names, loaded):
            np.testing.assert_allclose(loaded_arr, arrays[name])

    def test_extension_is_optional_and_never_doubled(self, tmp_path):
        """
        The archive is always .npz, so save_gyroid_matrices()/
        load_gyroid_matrices() should both accept a path with or without the
        extension: appending it automatically when missing, and not
        doubling it up (e.g. "foo.npz.npz") when the caller already
        included it. Both forms should be usable interchangeably.
        """
        arrays = _make_arrays()

        bare_path = tmp_path / "no_extension"
        io_ops.save_gyroid_matrices(str(bare_path), **arrays)
        assert bare_path.with_suffix(".npz").exists()

        explicit_path = tmp_path / "with_extension.npz"
        io_ops.save_gyroid_matrices(str(explicit_path), **arrays)
        assert explicit_path.exists()
        assert not (tmp_path / "with_extension.npz.npz").exists()

        # both should be loadable using either the bare or the .npz-suffixed form
        loaded_from_bare = io_ops.load_gyroid_matrices(str(bare_path))
        loaded_from_explicit = io_ops.load_gyroid_matrices(str(explicit_path))
        np.testing.assert_allclose(loaded_from_bare[0], arrays["Xres"])
        np.testing.assert_allclose(loaded_from_explicit[0], arrays["Xres"])

    def test_save_rejects_mismatched_shapes(self, tmp_path):
        """If any array's shape doesn't match the others (or can't be broadcast to match), save_gyroid_matrices() should raise ValueError and not write a file."""
        arrays = _make_arrays()
        arrays["thickness"] = np.ones((2, 2, 2))
        outfile = tmp_path / "should_not_exist.npz"
        with pytest.raises(ValueError):
            io_ops.save_gyroid_matrices(str(outfile), **arrays)
        assert not outfile.exists()

    def test_load_missing_file_raises_file_not_found_error(self, tmp_path):
        """Loading a path that doesn't exist should raise FileNotFoundError, not fail silently or return None."""
        with pytest.raises(FileNotFoundError):
            io_ops.load_gyroid_matrices(str(tmp_path / "nope.npz"))

    def test_load_corrupted_file_raises_runtime_error(self, tmp_path):
        """Loading a file that exists but isn't a valid .npz archive should raise RuntimeError (wrapping the underlying parsing exception), not fail silently or return None."""
        bad_file = tmp_path / "corrupted.npz"
        bad_file.write_bytes(b"not a real npz file")
        with pytest.raises(RuntimeError):
            io_ops.load_gyroid_matrices(str(bad_file))


# ============================================================================
# 2 - TestGyroidModelSaveLoadRoundtrip
# ============================================================================
class TestGyroidModelSaveLoadRoundtrip:
    """
    One end-to-end test tying io_ops back to GyroidModel.save()/.load(),
    since that's the path real users go through rather than calling io_ops
    directly. This is what would actually catch a regression if someone
    changed which fields GyroidModel.save() passes through to io_ops.
    """

    def test_model_roundtrip_preserves_field_and_coords(self, tmp_path):
        """
        GyroidModel.save() delegates to io_ops.save_gyroid_matrices() and
        .load() delegates to io_ops.load_gyroid_matrices(); this checks the
        whole chain end-to-end: compute a field, save it, load it into a new
        model, and confirm the field and coordinates match exactly. Mesh
        data (verts/faces) isn't part of the saved format, so it should come
        back None on the loaded copy, per GyroidModel.load()'s docstring.
        """
        gyroid_mod = pytest.importorskip("gyroid_utils.tpms_gyroid")
        lin = np.linspace(0, 2, 6)
        x, y, z = np.meshgrid(lin, lin, lin, indexing="ij")
        model = gyroid_mod.GyroidModel(x, y, z, 1.0, 1.0, 1.0, 0.2)
        model.compute_field(mode="abs")

        outfile = tmp_path / "model"
        model.save(str(outfile))

        loaded = gyroid_mod.GyroidModel.load(str(outfile))
        np.testing.assert_allclose(loaded.v, model.v)
        np.testing.assert_allclose(loaded.x, model.x)
        # mesh data is not persisted; a freshly loaded model has none
        assert loaded.verts is None and loaded.faces is None

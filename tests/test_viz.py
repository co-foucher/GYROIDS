"""
Tests for gyroid_utils.viz.
"""
import logging

import numpy as np
import pytest

#this is just like the import in gyroid_utils.viz, but we do it here so that pytest can skip all tests in this module if gyroid_utils isn't importable
viz = pytest.importorskip("gyroid_utils.viz")

"""
============================================================================
0 - helper functions
1 - TestSaveMeshAsHtml
2 - TestPlotHistogram
3 - TestTwodViewOfMatrix
4 - TestViewMesh
============================================================================
"""

# ============================================================================
# 0 - helper functions
# ============================================================================

def _make_triangle_mesh():
    """A single flat triangle: the simplest valid (verts, faces) input for these functions."""
    verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)
    faces = np.array([[0, 1, 2]])
    return verts, faces


# ============================================================================
# 1 - TestSaveMeshAsHtml
# ============================================================================
class TestSaveMeshAsHtml:
    """
    Tests for viz.save_mesh_as_html().

    These tests cover the None/empty guards (which raise TypeError/
    ValueError rather than silently no-op-ing), a real write to disk, and
    the colorscale-option precedence rule (defaults to "normal" if none or
    several of the four show_*_colorscale flags are True).
    """

    def test_none_faces_or_verts_raises_type_error(self, tmp_path):
        """faces=None (or verts=None) should raise TypeError and not write a file, per the explicit guard."""
        outfile = tmp_path / "preview"
        with pytest.raises(TypeError):
            viz.save_mesh_as_html(None, None, str(outfile))
        assert not outfile.with_suffix(".html").exists()

    def test_zero_faces_raises_value_error(self, tmp_path):
        """An empty faces array has nothing to render, so it should raise ValueError and not write a file."""
        verts, _ = _make_triangle_mesh()
        faces = np.zeros((0, 3), dtype=int)
        outfile = tmp_path / "preview"
        with pytest.raises(ValueError):
            viz.save_mesh_as_html(faces, verts, str(outfile))
        assert not outfile.with_suffix(".html").exists()

    def test_writes_html_file_for_valid_mesh(self, tmp_path):
        """A valid single-triangle mesh should produce a non-empty <file_name>.html containing real Plotly output."""
        verts, faces = _make_triangle_mesh()
        outfile = tmp_path / "preview"
        viz.save_mesh_as_html(faces, verts, str(outfile))

        html_path = outfile.with_suffix(".html")
        assert html_path.exists()
        content = html_path.read_text(encoding="utf-8")
        assert len(content) > 0
        assert "plotly" in content.lower()

    def test_defaults_to_normal_colorscale_when_none_selected(self, tmp_path, caplog):
        """If none of the four show_*_colorscale flags are True, it should log a warning and fall back to normal colorscale rather than silently render nothing."""
        verts, faces = _make_triangle_mesh()
        outfile = tmp_path / "preview"
        with caplog.at_level(logging.WARNING, logger="gyroid_utils"):
            viz.save_mesh_as_html(faces, verts, str(outfile))
        assert "Defaulting to normal colorscale" in caplog.text

    def test_defaults_to_normal_colorscale_when_multiple_selected(self, tmp_path, caplog):
        """If more than one show_*_colorscale flag is True, it should log a warning and fall back to normal colorscale rather than picking one arbitrarily."""
        verts, faces = _make_triangle_mesh()
        outfile = tmp_path / "preview"
        with caplog.at_level(logging.WARNING, logger="gyroid_utils"):
            viz.save_mesh_as_html(
                faces, verts, str(outfile),
                show_flat_colorscale=True,
                show_random_colorscale=True,
            )
        assert "Multiple colorscale options selected" in caplog.text


# ============================================================================
# 2 - TestPlotHistogram
# ============================================================================
class TestPlotHistogram:
    """
    Tests for viz.plot_histogram().

    Test the None/empty guards directly (they raise ValueError before any
    Plotly call is reached). The success path is still exercised, but with
    Figure.show monkeypatched to a no-op, so the histogram computation and
    figure-building code run for real without anything actually trying to
    display.
    """

    def test_none_input_raises_value_error(self):
        """face_areas=None should raise ValueError, without touching Plotly at all."""
        with pytest.raises(ValueError):
            viz.plot_histogram(None)

    def test_empty_array_raises_value_error(self):
        """An empty face_areas array should raise ValueError, without touching Plotly at all."""
        with pytest.raises(ValueError):
            viz.plot_histogram(np.array([]))

    def test_valid_input_builds_histogram_without_raising(self, monkeypatch):
        """
        With Figure.show patched to a no-op, a real array of areas should
        make it through np.histogram() and the figure-building code without
        raising, confirming that part of the pipeline is sound.
        """
        import plotly.graph_objects as go
        monkeypatch.setattr(go.Figure, "show", lambda self, *a, **k: None)

        areas = np.array([0.1, 0.2, 0.15, 0.3, 0.25])
        viz.plot_histogram(areas, BINS=5)


# ============================================================================
# 3 - TestTwodViewOfMatrix
# ============================================================================
class TestTwodViewOfMatrix:
    """
    Tests for viz.twod_view_of_matrix().

    Same reasoning as TestPlotHistogram: the success path ends in
    fig.show(), so it's only exercised with Figure.show monkeypatched to a
    no-op. The two validation guards (wrong ndim, mismatched grid shape)
    are safe to test directly since they raise ValueError before any Plotly
    call is reached.
    """

    def test_wrong_ndim_raises_value_error(self):
        """v must be 3D; a 2D array should raise ValueError before any Plotly call."""
        v = np.zeros((4, 4))
        with pytest.raises(ValueError):
            viz.twod_view_of_matrix(v)

    def test_mismatched_grid_shape_raises_value_error(self):
        """x/y/z grids that don't match v's shape should raise ValueError before any Plotly call."""
        v = np.zeros((4, 4, 4))
        x = np.zeros((2, 2, 2))
        y = np.zeros((2, 2, 2))
        z = np.zeros((2, 2, 2))
        with pytest.raises(ValueError):
            viz.twod_view_of_matrix(v, x=x, y=y, z=z)

    def test_valid_input_builds_frames_without_raising(self, monkeypatch):
        """
        With Figure.show patched to a no-op, a small valid field should make
        it through the per-slice frame-building loop and slider/heatmap
        construction without raising.
        """
        import plotly.graph_objects as go
        monkeypatch.setattr(go.Figure, "show", lambda self, *a, **k: None)

        v = np.random.default_rng(0).random((3, 3, 2))
        viz.twod_view_of_matrix(v)


# ============================================================================
# 4 - TestViewMesh
# ============================================================================
class TestViewMesh:
    """
    Tests for viz.view_mesh(), a thin wrapper around
    save_mesh_as_html(..., save=False).

    Its success path ends in fig.show() (via save_mesh_as_html's save=False
    branch), so - as with the two classes above - it's only exercised with
    Figure.show monkeypatched to a no-op. The None-input case is safe to
    test directly, since save_mesh_as_html's None guard raises before the
    save flag ever matters.
    """

    def test_none_input_raises_type_error(self):
        """faces=None/verts=None should raise TypeError, via save_mesh_as_html's guard, before the save=False branch is ever reached."""
        with pytest.raises(TypeError):
            viz.view_mesh(None, None)

    def test_valid_mesh_displays_without_raising(self, monkeypatch):
        """With Figure.show patched to a no-op, a valid single-triangle mesh should build and 'display' without raising."""
        import plotly.graph_objects as go
        monkeypatch.setattr(go.Figure, "show", lambda self, *a, **k: None)

        verts, faces = _make_triangle_mesh()
        viz.view_mesh(faces, verts)

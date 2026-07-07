"""
Parametrized tests for every concrete TPMS surface subclass of TPMSModel:
GyroidModel, SchwartzPModel, DiamondModel, IWPModel, NeoviusModel,
FischerKochSModel, FRDModel, LidinoidModel, SplitPModel.

Validation, guard rails, and the abs/signed/distance field pipeline itself
all live in TPMSModel and are already tested once, generically, through a
dummy subclass in test_tpms_base.py. What's specific to each of these real
classes is just its _surface_term() formula and its DEFAULT_FIELD_MODE, so
that's all this file checks - once per class, via parametrize, instead of
copy-pasting a near-identical test file per surface.
"""
import numpy as np
import pytest

#one importorskip per module, same convention as the other test files: if gyroid_utils isn't importable, the first of these skips the whole file
gyroid_mod = pytest.importorskip("gyroid_utils.TPMS_classes.tpms_gyroid")
schwartzp_mod = pytest.importorskip("gyroid_utils.TPMS_classes.tpms_schwartzp")
diamond_mod = pytest.importorskip("gyroid_utils.TPMS_classes.tpms_diamond")
iwp_mod = pytest.importorskip("gyroid_utils.TPMS_classes.tpms_iwp")
neovius_mod = pytest.importorskip("gyroid_utils.TPMS_classes.tpms_neovius")
fischerkochs_mod = pytest.importorskip("gyroid_utils.TPMS_classes.tpms_fischerkochs")
frd_mod = pytest.importorskip("gyroid_utils.TPMS_classes.tpms_frd")
lidinoid_mod = pytest.importorskip("gyroid_utils.TPMS_classes.tpms_lidinoid")
splitp_mod = pytest.importorskip("gyroid_utils.TPMS_classes.tpms_splitp")

"""
============================================================================
0 - independent formula reimplementations (one per surface)
1 - _CASES : (model class, formula fn, DEFAULT_FIELD_MODE) per surface
2 - TestTPMSSurfaceFormulas (parametrized across every case in _CASES)
============================================================================
"""

# ============================================================================
# 0 - independent formula reimplementations (one per surface)
# ============================================================================
# Each of these is a hand-written copy of that surface's _surface_term(),
# taken from its docstring/NOTES section, so compute_field() gets checked
# against an independent implementation rather than against itself.

def _gyroid_term(x, y, z, px, py, pz):
    return (
        np.sin((2 * np.pi / px) * x) * np.cos((2 * np.pi / py) * y)
        + np.sin((2 * np.pi / py) * y) * np.cos((2 * np.pi / pz) * z)
        + np.sin((2 * np.pi / pz) * z) * np.cos((2 * np.pi / px) * x)
    )


def _schwartzp_term(x, y, z, px, py, pz):
    return (
        np.cos((2 * np.pi / px) * x)
        + np.cos((2 * np.pi / py) * y)
        + np.cos((2 * np.pi / pz) * z)
    )


def _diamond_term(x, y, z, px, py, pz):
    X, Y, Z = (2 * np.pi / px) * x, (2 * np.pi / py) * y, (2 * np.pi / pz) * z
    return (
        np.sin(X) * np.sin(Y) * np.sin(Z)
        + np.sin(X) * np.cos(Y) * np.cos(Z)
        + np.cos(X) * np.sin(Y) * np.cos(Z)
        + np.cos(X) * np.cos(Y) * np.sin(Z)
    )


def _iwp_term(x, y, z, px, py, pz):
    X, Y, Z = (2 * np.pi / px) * x, (2 * np.pi / py) * y, (2 * np.pi / pz) * z
    return (
        2 * (np.cos(X) * np.cos(Y) + np.cos(Y) * np.cos(Z) + np.cos(Z) * np.cos(X))
        - (np.cos(2 * X) + np.cos(2 * Y) + np.cos(2 * Z))
    )


def _neovius_term(x, y, z, px, py, pz):
    X, Y, Z = (2 * np.pi / px) * x, (2 * np.pi / py) * y, (2 * np.pi / pz) * z
    return (
        3 * (np.cos(X) + np.cos(Y) + np.cos(Z))
        + 4 * np.cos(X) * np.cos(Y) * np.cos(Z)
    )


def _fischerkochs_term(x, y, z, px, py, pz):
    X, Y, Z = (2 * np.pi / px) * x, (2 * np.pi / py) * y, (2 * np.pi / pz) * z
    return (
        np.cos(2 * X) * np.sin(Y) * np.cos(Z)
        + np.cos(2 * Y) * np.sin(Z) * np.cos(X)
        + np.cos(2 * Z) * np.sin(X) * np.cos(Y)
    )


def _frd_term(x, y, z, px, py, pz):
    X, Y, Z = (2 * np.pi / px) * x, (2 * np.pi / py) * y, (2 * np.pi / pz) * z
    return (
        8 * np.cos(X) * np.cos(Y) * np.cos(Z)
        + np.cos(2 * X) * np.cos(2 * Y) * np.cos(2 * Z)
        - (np.cos(2 * X) * np.cos(2 * Y) + np.cos(2 * Y) * np.cos(2 * Z) + np.cos(2 * Z) * np.cos(2 * X))
    )


def _lidinoid_term(x, y, z, px, py, pz):
    X, Y, Z = (2 * np.pi / px) * x, (2 * np.pi / py) * y, (2 * np.pi / pz) * z
    return (
        np.sin(2 * X) * np.cos(Y) * np.sin(Z)
        + np.sin(2 * Y) * np.cos(Z) * np.sin(X)
        + np.sin(2 * Z) * np.cos(X) * np.sin(Y)
        - np.cos(2 * X) * np.cos(2 * Y)
        - np.cos(2 * Y) * np.cos(2 * Z)
        - np.cos(2 * Z) * np.cos(2 * X)
        + 0.3
    )


def _splitp_term(x, y, z, px, py, pz):
    X, Y, Z = (2 * np.pi / px) * x, (2 * np.pi / py) * y, (2 * np.pi / pz) * z
    return (
        1.1 * (np.sin(2 * X) * np.sin(Z) * np.cos(Y)
               + np.sin(2 * Y) * np.sin(X) * np.cos(Z)
               + np.sin(2 * Z) * np.sin(Y) * np.cos(X))
        - 0.2 * (np.cos(2 * X) * np.cos(2 * Y) + np.cos(2 * Y) * np.cos(2 * Z) + np.cos(2 * Z) * np.cos(2 * X))
        - 0.4 * (np.cos(2 * X) + np.cos(2 * Y) + np.cos(2 * Z))
    )


# ============================================================================
# 1 - _CASES
# ============================================================================
# (model class, independent formula fn, expected DEFAULT_FIELD_MODE), one
# entry per concrete TPMS surface. Adding a 10th surface later is just a
# new formula function above and a new line here.
_CASES = [
    pytest.param(gyroid_mod.GyroidModel, _gyroid_term, "distance", id="gyroid"),
    pytest.param(schwartzp_mod.SchwartzPModel, _schwartzp_term, "abs", id="schwartzp"),
    pytest.param(diamond_mod.DiamondModel, _diamond_term, "distance", id="diamond"),
    pytest.param(iwp_mod.IWPModel, _iwp_term, "distance", id="iwp"),
    pytest.param(neovius_mod.NeoviusModel, _neovius_term, "distance", id="neovius"),
    pytest.param(fischerkochs_mod.FischerKochSModel, _fischerkochs_term, "distance", id="fischerkochs"),
    pytest.param(frd_mod.FRDModel, _frd_term, "distance", id="frd"),
    pytest.param(lidinoid_mod.LidinoidModel, _lidinoid_term, "distance", id="lidinoid"),
    pytest.param(splitp_mod.SplitPModel, _splitp_term, "distance", id="splitp"),
]


# ============================================================================
# 2 - TestTPMSSurfaceFormulas
# ============================================================================
@pytest.mark.parametrize("model_cls, term_fn, default_mode", _CASES)
class TestTPMSSurfaceFormulas:
    """
    Runs the same three checks against every concrete TPMS surface class,
    each parametrized with its own independently-written formula function
    and expected DEFAULT_FIELD_MODE (from _CASES above). If a future edit
    to any one surface's trig formula, a sign flip, a period substitution,
    or its default mode changes, the corresponding parametrized case fails
    - without needing a dedicated test file for that surface.
    """

    def test_abs_mode_matches_formula(self, small_grid, model_cls, term_fn, default_mode):
        """mode="abs" should equal thickness - |term|, per TPMSModel's docstring, for every surface."""
        x, y, z = small_grid
        model = model_cls(x, y, z, 1.3, 1.1, 0.9, 0.25)
        v = model.compute_field(mode="abs")
        expected = 0.25 - np.abs(term_fn(x, y, z, 1.3, 1.1, 0.9))
        np.testing.assert_allclose(v, expected)

    def test_signed_mode_matches_formula(self, small_grid, model_cls, term_fn, default_mode):
        """mode="signed" should equal term - thickness (a plain level-set, no abs()), for every surface."""
        x, y, z = small_grid
        model = model_cls(x, y, z, 1.3, 1.1, 0.9, 0.25)
        v = model.compute_field(mode="signed")
        expected = term_fn(x, y, z, 1.3, 1.1, 0.9) - 0.25
        np.testing.assert_allclose(v, expected)

    def test_default_field_mode_matches_class_attribute(self, small_grid, model_cls, term_fn, default_mode):
        """Each class's DEFAULT_FIELD_MODE should match what's documented (_CASES above), and compute_field() with no mode argument should behave exactly like passing that mode explicitly."""
        x, y, z = small_grid
        model = model_cls(x, y, z, 1.3, 1.1, 0.9, 0.25)
        assert model.DEFAULT_FIELD_MODE == default_mode

        v_default = model.compute_field()
        v_explicit = model.compute_field(mode=default_mode)
        np.testing.assert_allclose(v_default, v_explicit)

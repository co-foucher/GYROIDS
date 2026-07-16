import numpy as np

from .tpms_base import TPMSModel, create_a_tpms


"""
#=====================================================================================================================
0 - (reserved)
1 - CustomTPMSModel (class)
2 - CustomTPMSModel._surface_term
3 - create_a_custom_tpms
#=====================================================================================================================

NOTE ON THIS MODULE
--------------------
This generalizes tpms_gyroid.py / tpms_schwartzp.py / etc.: instead of a
hardcoded `_surface_term()`, CustomTPMSModel returns a *precomputed*
implicit-surface field array supplied at construction time. Everything
else - field-mode processing, meshing, simplification, export, previews -
is inherited unchanged from TPMSModel, exactly like the built-in types.

This class has no notion of equation strings, sympy, or parsing of any
kind - it only ever deals with plain numpy arrays, same as every other
TPMSModel subclass. Turning a user-typed formula (e.g. from the GUI's
"Generate TPMS" page) into a numeric array is entirely the caller's
responsibility; see app/components/equation_input.py (parsing,
validation, the widget, and the evaluation call site all live there) for
the app-side implementation that produces the `field`/`thickness` arrays
this class expects. That separation is deliberate: the library stays a
plain numerical-computation package, and anything GUI-specific lives in
the application layer instead.

NOTE on px/py/pz: unlike the built-in types (GyroidModel etc.), where
px/py/pz are the periods plugged into a fixed, hardcoded equation, here
the field is already fully defined by the `field` array the caller
supplies - there's nothing left for a separate period to parameterize.
So CustomTPMSModel itself simply doesn't take px/py/pz at all - __init__
passes fixed placeholder values (1.0) to TPMSModel.__init__ on the
caller's behalf, since that base constructor still requires them
positionally (it stores/saves them like every other parameter).

create_a_custom_tpms() below still delegates to the shared
tpms_base.create_a_tpms() pipeline, same as create_a_gyroid() etc. - that
helper calls every model type uniformly as
model_cls(x, y, z, px, py, pz, thickness). To bridge that uniform
convention to CustomTPMSModel's px/py/pz-less constructor, it passes a
small wrapper as model_cls that accepts (and discards) px/py/pz, and
always drives px=py=pz=1.0 into create_a_tpms() itself - matching
CustomTPMSModel's own placeholder values above.
"""


# =====================================================================
# 1) CustomTPMSModel
# =====================================================================
class CustomTPMSModel(TPMSModel):
    """
    ============================================================================
    1) CUSTOMTPMSMODEL
    TPMS model whose implicit surface is supplied directly as a
    precomputed scalar field array, instead of being computed from a
    hardcoded formula like GyroidModel/SchwartzPModel/etc. Same API as
    every other TPMSModel subclass otherwise (compute_field, generate_mesh,
    simplify_mesh, ...), except it takes no px/py/pz (see module NOTE).
    ============================================================================

    PARAMETERS
    ----------
    x, y, z, thickness : see TPMSModel. thickness must already be a plain
        scalar or numpy array (same shape as x) - this class does no
        parsing of its own; see app/components/equation_input.py if the
        caller is starting from a formula string instead of a plain array.
    field : np.ndarray
        Precomputed implicit surface values F(x, y, z), same shape as x.
        Returned as-is by _surface_term().

    RAISES
    ------
    ValueError
        If `field`'s shape doesn't match x's shape.

    NOTES
    -----
    - Reloading via `TPMSModel.load()` restores the saved field/params
      (the field itself is fully valid) but obviously has no formula to
      "re-show" - there never was one at this layer.

    EXAMPLE
    -------
    >>> field = np.sin(x) + np.cos(y) + np.sin(z)  # or however the caller computed it
    >>> model = CustomTPMSModel(x, y, z, 0.2, field=field)
    >>> field_out = model.compute_field()
    """

    DEFAULT_FIELD_MODE = "distance"

    def __init__(self, x, y, z, thickness, field: np.ndarray):
        field = np.asarray(field, dtype=float)
        x_shape = np.asarray(x).shape
        if field.shape != x_shape:
            raise ValueError(
                f"CustomTPMSModel: field shape {field.shape} does not match "
                f"x/y/z shape {x_shape}."
            )
        self._field = field

        # TPMSModel.__init__ requires px/py/pz positionally (it stores and
        # later saves them, like every other parameter) even though this
        # subclass has no use for them - 1.0 is an arbitrary placeholder,
        # never read back by anything that affects the field.
        super().__init__(x, y, z, 1.0, 1.0, 1.0, thickness)

    # =====================================================================
    # 2) _surface_term
    # =====================================================================
    def _surface_term(self) -> np.ndarray:
        """
        ============================================================================
        2) _SURFACE_TERM
        Returns the field array supplied at construction time, unchanged.
        ============================================================================

        RETURNS
        -------
        term : np.ndarray
            F(x, y, z), same shape as self.x.
        """
        return self._field


# =====================================================================
# 3) create_a_custom_tpms
# =====================================================================
def create_a_custom_tpms(x: np.ndarray,
                         y: np.ndarray,
                         z: np.ndarray,
                         t: np.ndarray,
                         field: np.ndarray,
                         save_path: str,
                         baseplate_thickness: float = 0.0,
                         step_size: int = 2,
                         simplification_factor=0.9,
                         field_mode: str = None):
    """
    ============================================================================
    3) CREATE_A_CUSTOM_TPMS
    Convenience function mirroring create_a_gyroid()/create_a_diamond()/etc.
    for the precomputed-field case. Delegates to the shared
    tpms_base.create_a_tpms() pipeline, same as every built-in type - see
    the module-level NOTE for how px/py/pz (always 1.0, unused by
    CustomTPMSModel) are bridged across that shared helper's uniform
    calling convention.
    ============================================================================

    PARAMETERS
    ----------
    x, y, z, t : see create_a_gyroid.
    field : np.ndarray
        Precomputed implicit surface values (see CustomTPMSModel).
    save_path, baseplate_thickness, step_size, simplification_factor,
    field_mode : see create_a_gyroid.

    RETURNS
    -------
    success : bool
        True if a valid mesh was generated and exported, False otherwise.
    """
    # create_a_tpms() instantiates every model type uniformly as
    # model_cls(x, y, z, px, py, pz, thickness). CustomTPMSModel doesn't
    # take px/py/pz (see module NOTE), so this thin wrapper absorbs and
    # discards them before forwarding to the real constructor.
    def _model_cls(x, y, z, px, py, pz, thickness):
        return CustomTPMSModel(x, y, z, thickness, field=field)

    return create_a_tpms(
        _model_cls,
        x, y, z, 1.0, 1.0, 1.0, t,
        save_path,
        baseplate_thickness=baseplate_thickness,
        step_size=step_size,
        simplification_factor=simplification_factor,
        field_mode=field_mode,
    )

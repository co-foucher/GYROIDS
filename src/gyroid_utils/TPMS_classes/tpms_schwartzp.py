import numpy as np

# Use relative imports so this module works when used as a package
from .tpms_base import TPMSModel, create_a_tpms


"""
#=====================================================================================================================
0 - (reserved)
1 - SchwartzPModel (class)
2 - SchwartzPModel._surface_term
3 - create_a_schwartz_p
#=====================================================================================================================

This is the Schwartz P TPMS type, refactored on top of TPMSModel (see
tpms_base.py). All the grid validation / field pipeline / meshing / export
logic lives in the base class; this subclass only supplies the Schwartz P's
implicit surface equation.

# =====================================================================================================================
USE EXAMPLE
-----------
>>> model = SchwartzPModel(x, y, z, px, py, pz, thickness)
>>> field = model.compute_field()
"""


# =====================================================================
# 1) SchwartzPModel
# =====================================================================
class SchwartzPModel(TPMSModel):
    """
    ============================================================================
    1) SCHWARTZPMODEL
    Represents a Schwartz P (Primitive) TPMS scalar field defined on a 3D grid.
    ============================================================================

    PARAMETERS
    ----------
    x, y, z : np.ndarray
        Numpy arrays of identical shape describing coordinates for the field.
    px, py, pz : float or np.ndarray
        Periods (mm). Each may be a scalar (most common) or an array with the
        same shape as x/y/z (for per-voxel period variations).
    thickness : float or np.ndarray
        Scalar or array (shape identical to x/y/z) controlling the isosurface
        threshold.

    NOTES
    -----
    - The Schwartz P surface is one of the simplest triply-periodic minimal
      surfaces. Its implicit equation is:

          F(x, y, z) = cos(2π/px · x) + cos(2π/py · y) + cos(2π/pz · z) = 0

      The field amplitude range is [-3, +3] (each cosine term spans [-1, 1]).
      Because of its cubic symmetry and mirror symmetry (unlike the gyroid),
      it is easy to tile and has well-characterised mechanical properties.
    - `load` is inherited from TPMSModel and returns a SchwartzPModel instance.

    EXAMPLE
    -------
    >>> model = SchwartzPModel(x, y, z, px, py, pz, thickness)
    >>> field = model.compute_field()
    >>> model = SchwartzPModel.load("gyroid_data.npz")
    """

    # Matches the original SchwartzP.py behavior: compute_field() with no
    # mode defaults to "abs".
    DEFAULT_FIELD_MODE = "abs"

    # =====================================================================
    # 2) _surface_term
    # =====================================================================
    def _surface_term(self) -> np.ndarray:
        """
        ============================================================================
        2) _SURFACE_TERM
        Schwartz P implicit surface:
            F = cos(2π/px·x) + cos(2π/py·y) + cos(2π/pz·z)
        Amplitude range: [-3, +3] (three cosine terms each in [-1, 1]).
        ============================================================================

        RETURNS
        -------
        term : np.ndarray
            F(x, y, z), same shape as self.x.
        """
        return (
            np.cos((2 * np.pi / self.px) * self.x)
            + np.cos((2 * np.pi / self.py) * self.y)
            + np.cos((2 * np.pi / self.pz) * self.z)
        )


# =====================================================================
# 3) create_a_schwartz_p
# =====================================================================
def create_a_schwartz_p(x: np.ndarray,
                        y: np.ndarray,
                        z: np.ndarray,
                        px: np.ndarray,
                        py: np.ndarray,
                        pz: np.ndarray,
                        t: np.ndarray,
                        save_path: str,
                        baseplate_thickness: float = 0.0,
                        step_size: int = 2,
                        simplification_factor=0.9,
                        field_mode: str = "distance"):
    """
    ============================================================================
    3) CREATE_A_SCHWARTZ_P
    Convenience function to create a Schwartz P model, compute the field,
    generate and simplify the mesh, and save results. Thin wrapper around
    tpms_base.create_a_tpms(SchwartzPModel, ...).
    ============================================================================

    PARAMETERS
    ----------
    x, y, z : np.ndarray
        Coordinate grids (3D arrays of identical shape).
    px, py, pz : np.ndarray
        Periods in mm (scalars or arrays matching x/y/z shape).
    t : np.ndarray
        Thickness parameter (scalar or array matching x/y/z shape). In
        'distance' mode this is a physical wall thickness in mm. In 'abs'
        mode this is a dimensionless threshold on |F| (range 0-3).
    save_path : str
        Base path for saving the .stl mesh and HTML preview (without extension).
    baseplate_thickness : float, optional
        Thickness of the baseplates to add at ±Z. 0 = none (default).
    step_size : int, optional
        Marching cubes step size (higher = faster but less detailed mesh,
        default = 2).
    simplification_factor : float, optional
        Target fraction of faces to keep during simplification (0.5 = keep
        50% of faces), or target number of faces if >1 (e.g. 10000). Default = 0.9.
    field_mode : str, optional
        One of 'distance' (recommended, default), 'abs', 'signed', or
        'distance_fast'.

    RETURNS
    -------
    success : bool
        True if a valid mesh was generated and exported, False otherwise.
    """
    return create_a_tpms(
        SchwartzPModel,
        x, y, z, px, py, pz, t,
        save_path,
        baseplate_thickness=baseplate_thickness,
        step_size=step_size,
        simplification_factor=simplification_factor,
        field_mode=field_mode,
    )

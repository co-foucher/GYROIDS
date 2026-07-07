import numpy as np

# Use relative imports so this module works when used as a package
from .tpms_base import TPMSModel, create_a_tpms


"""
#=====================================================================================================================
0 - (reserved)
1 - LidinoidModel (class)
2 - LidinoidModel._surface_term
3 - create_a_lidinoid
#=====================================================================================================================

This is the Lidinoid TPMS type, built on top of TPMSModel (see
tpms_base.py). All the grid validation / field pipeline / meshing / export
logic lives in the base class; this subclass only supplies the Lidinoid's
implicit surface equation.

# =====================================================================================================================
USE EXAMPLE
-----------
>>> model = LidinoidModel(x, y, z, px, py, pz, thickness)
>>> field = model.compute_field()
"""


# =====================================================================
# 1) LidinoidModel
# =====================================================================
class LidinoidModel(TPMSModel):
    """
    ============================================================================
    1) LIDINOIDMODEL
    Represents a Lidinoid (L) TPMS scalar field defined on a 3D grid.
    ============================================================================

    PARAMETERS
    ----------
    x, y, z : np.ndarray
        Numpy arrays of identical shape describing coordinates for the field.
    px, py, pz : float or np.ndarray
        Periods. Each may be a scalar (most common) or an array with the
        same shape as x/y/z (for per-voxel period variations).
    thickness : float or np.ndarray
        Scalar or array (shape identical to x/y/z) controlling the isosurface
        threshold.

    NOTES
    -----
    - The Lidinoid surface has no inherent mirror symmetry (unlike Primitive
      or Diamond), giving it a chiral, gyroid-like topology. Its implicit
      equation is:

          F(x, y, z) = sin(2X)cos(Y)sin(Z) + sin(2Y)cos(Z)sin(X)
                     + sin(2Z)cos(X)sin(Y)
                     - cos(2X)cos(2Y) - cos(2Y)cos(2Z) - cos(2Z)cos(2X) + 0.3

      where X = 2π/px·x, Y = 2π/py·y, Z = 2π/pz·z. The "+ 0.3" offset shifts
      the zero-isosurface to give the Lidinoid its characteristic asymmetric
      channels; the field amplitude range is roughly [-2.7, +2.3].
    - `load` is inherited from TPMSModel and returns a LidinoidModel instance.

    EXAMPLE
    -------
    >>> model = LidinoidModel(x, y, z, px, py, pz, thickness)
    >>> field = model.compute_field()
    >>> model = LidinoidModel.load("gyroid_data.npz")
    """

    # Same convention as GyroidModel: compute_field() with no mode defaults
    # to "distance".
    DEFAULT_FIELD_MODE = "distance"

    # =====================================================================
    # 2) _surface_term
    # =====================================================================
    def _surface_term(self) -> np.ndarray:
        """
        ============================================================================
        2) _SURFACE_TERM
        Lidinoid implicit surface:
            F = sin(2X)cos(Y)sin(Z) + sin(2Y)cos(Z)sin(X) + sin(2Z)cos(X)sin(Y)
              - cos(2X)cos(2Y) - cos(2Y)cos(2Z) - cos(2Z)cos(2X) + 0.3
        Amplitude range: roughly [-2.7, +2.3].
        ============================================================================

        RETURNS
        -------
        term : np.ndarray
            F(x, y, z), same shape as self.x.
        """
        X = (2 * np.pi / self.px) * self.x
        Y = (2 * np.pi / self.py) * self.y
        Z = (2 * np.pi / self.pz) * self.z
        return (
            np.sin(2 * X) * np.cos(Y) * np.sin(Z)
            + np.sin(2 * Y) * np.cos(Z) * np.sin(X)
            + np.sin(2 * Z) * np.cos(X) * np.sin(Y)
            - np.cos(2 * X) * np.cos(2 * Y)
            - np.cos(2 * Y) * np.cos(2 * Z)
            - np.cos(2 * Z) * np.cos(2 * X)
            + 0.3
        )


# =====================================================================
# 3) create_a_lidinoid
# =====================================================================
def create_a_lidinoid(x: np.ndarray,
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
    3) CREATE_A_LIDINOID
    Convenience function to create a Lidinoid model, compute the field,
    generate and simplify the mesh, and save results. Thin wrapper around
    tpms_base.create_a_tpms(LidinoidModel, ...).
    ============================================================================

    PARAMETERS
    ----------
    x, y, z : np.ndarray
        Coordinate grids (3D arrays of identical shape).
    px, py, pz : np.ndarray
        Periods (scalars or arrays matching x/y/z shape).
    t : np.ndarray
        Thickness parameter (scalar or array matching x/y/z shape).
    save_path : str
        Base path for saving the .stl mesh and HTML preview (without extension).
    baseplate_thickness : float, optional
        Thickness of the baseplates to add. 0 = none (default).
    step_size : int, optional
        Marching cubes step size (higher = faster but less detailed mesh, default = 2).
    simplification_factor : float, optional
        Target fraction of faces to keep during simplification (0.5 = keep
        50% of faces), or target number of faces if >1 (e.g. 10000). Default = 0.9.
    field_mode : str, optional
        Field computation mode passed to LidinoidModel.compute_field
        (default = "distance").

    RETURNS
    -------
    success : bool
        True if a valid mesh was generated and exported, False otherwise.
    """
    return create_a_tpms(
        LidinoidModel,
        x, y, z, px, py, pz, t,
        save_path,
        baseplate_thickness=baseplate_thickness,
        step_size=step_size,
        simplification_factor=simplification_factor,
        field_mode=field_mode,
    )

import numpy as np
from typing import Optional, Tuple, Union

# Use relative imports so this module works when used as a package
from . import io_ops, mesh_tools, viz
from .logger import logger


"""
#=====================================================================================================================
0 - (reserved)
1 - GyroidModel (class)
2 - GyroidModel.__init__
3 - GyroidModel._validate_inputs
4 - GyroidModel.compute_field
5 - GyroidModel.save
6 - GyroidModel.load
7 - GyroidModel.generate_mesh
8 - GyroidModel.simplify_mesh
9 - GyroidModel.export_stl
10 - GyroidModel.save_mesh_preview
11 - GyroidModel.check_mesh_quality
12 - GyroidModel.keep_largest_connected_component
13 - GyroidModel.add_baseplates
14 - GyroidModel.fix_mesh
15 - GyroidModel.smooth_mesh
16 - create_a_gyroid
#=====================================================================================================================
"""


# =====================================================================
# 1) GyroidModel
# =====================================================================
class GyroidModel:
    """
    ============================================================================
    1) GYROIDMODEL
    Represents a gyroid scalar field defined on a 3D grid and provides helpers
    to compute the field, generate/simplify a mesh, export results and run
    simple checks/visualizations.
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
    - `load` is a classmethod that loads saved gyroid parameters and field
      from disk.

    EXAMPLE
    -------
    >>> model = GyroidModel(x, y, z, px, py, pz, thickness)
    >>> field = model.compute_field()
    >>> model = GyroidModel.load("gyroid_data.npz")
    """

    # =====================================================================
    # 2) __init__
    # =====================================================================
    def __init__(
        self,
        x: np.ndarray,                              # x,y,z coordinates of the grid. 3D arrays of identical shape.
        y: np.ndarray,
        z: np.ndarray,
        px: Union[float, np.ndarray],               # periods in x,y,z directions. Scalars or arrays matching x/y/z shape.
        py: Union[float, np.ndarray],
        pz: Union[float, np.ndarray],
        thickness: Union[float, np.ndarray],        # thickness parameter. Scalar or array matching x/y/z shape.
        ):
        """
        ============================================================================
        2) __INIT__
        Initializes a GyroidModel from coordinate grids and gyroid parameters.
        ============================================================================

        PARAMETERS
        ----------
        x, y, z : np.ndarray
            3D coordinate arrays of identical shape.
        px, py, pz : float or np.ndarray
            Periods in x, y, z directions. Scalars or arrays matching x/y/z shape.
        thickness : float or np.ndarray
            Thickness parameter. Scalar or array matching x/y/z shape.

        RETURNS
        -------
        None
        """

        # --- needed data to create the object ---
        # Coordinate grids and parameters
        self.x = x
        self.y = y
        self.z = z
        self.px = px
        self.py = py
        self.pz = pz
        self.thickness = thickness

        # --- optional data to the object ---
        # Scalar field (computed by compute_field)
        self.v: Optional[np.ndarray] = None

        # Mesh data (filled by generate_mesh)
        self.verts: Optional[np.ndarray] = None
        self.faces: Optional[np.ndarray] = None

        self._validate_inputs()

    # =====================================================================
    # 3) _validate_inputs
    # =====================================================================
    def _validate_inputs(self):
        """
        ============================================================================
        3) _VALIDATE_INPUTS
        Validates shapes/types of inputs.
        ============================================================================

        PARAMETERS
        ----------
        None (operates on self.x, self.y, self.z, self.px, self.py, self.pz,
        self.thickness)

        RETURNS
        -------
        None

        NOTES
        -----
        - x, y, z must be numpy arrays of identical shape.
        - px/py/pz and thickness may be scalars or arrays that match x.shape.
        """
        if not (isinstance(self.x, np.ndarray) and isinstance(self.y, np.ndarray) and isinstance(self.z, np.ndarray)):
            raise TypeError("x, y, z must be numpy arrays.")

        if not (self.x.shape == self.y.shape == self.z.shape):
            raise ValueError("x, y, z must have identical shapes.")

        # helper to validate either scalar or same-shaped array
        def _check_param(name, val):            #note: _ is a convention o show which functions, methods and variables are internal to a class or module
            if np.isscalar(val):
                return
            if not isinstance(val, np.ndarray):
                raise TypeError(f"{name} must be a scalar or numpy array.")
            if val.shape != self.x.shape:
                raise ValueError(f"{name} array must have same shape as x/y/z.")

        _check_param("px", self.px)
        _check_param("py", self.py)
        _check_param("pz", self.pz)
        _check_param("thickness", self.thickness)

    # =====================================================================
    # 4) compute_field
    # =====================================================================
    def compute_field(self,
                      mode: str = "abs") -> np.ndarray:
        """
        ============================================================================
        4) COMPUTE_FIELD
        Computes the gyroid scalar field.
        ============================================================================

        PARAMETERS
        ----------
        mode : str, optional
            - "abs" (default): original behavior -> v = thickness - |term|
              (useful for value-space wall thresholding; thickness here is in
              term units).
            - "signed": standard level-set -> v = term - level (signed field).
            - "distance": produce a signed-distance-derived thickness field:
                1) binary = term > level
                2) compute signed distance (uses spacing)
                3) if physical_thickness provided (scalar or array matching
                   grid), v = physical_thickness/2 - |signed_dist| (positive
                   inside the desired wall band)

        RETURNS
        -------
        v : np.ndarray
            The computed scalar field (also stored in self.v).

        NOTES
        -----
        - spacing: voxel spacing used when computing distance transform (only
          for mode="distance").
        - physical_thickness: desired wall thickness in spatial units (only
          used for "distance" mode). May be a scalar or an ndarray with the
          same shape as x/y/z.
        """
        term = (
            np.sin((2 * np.pi / self.px) * self.x) * np.cos((2 * np.pi / self.py) * self.y)
            + np.sin((2 * np.pi / self.py) * self.y) * np.cos((2 * np.pi / self.pz) * self.z)
            + np.sin((2 * np.pi / self.pz) * self.z) * np.cos((2 * np.pi / self.px) * self.x)
        )

        if mode == "abs":
            # original behaviour: thickness interpreted in term-value units (supports scalar or per-voxel thickness)
            logger.info(f"Computing absolute field")
            self.v = self.thickness - np.abs(term)
            return self.v

        if mode == "signed":
            # signed level-set relative to provided level (C)
            logger.info(f"Computing signed field")
            self.v = term - self.thickness
            return self.v

        if mode == "distance" or mode == "distance_fast":
            # requires scipy
            logger.info(f"Computing distance field")
            # Auto-compute actual voxel spacing from the coordinate grids
            dx = float(self.x[1, 0, 0] - self.x[0, 0, 0]) if self.x.shape[0] > 1 else 1.0
            dy = float(self.y[0, 1, 0] - self.y[0, 0, 0]) if self.y.shape[1] > 1 else 1.0
            dz = float(self.z[0, 0, 1] - self.z[0, 0, 0]) if self.z.shape[2] > 1 else 1.0
            spacing = (dx, dy, dz)
            try:
                from scipy.ndimage import distance_transform_edt, distance_transform_cdt
            except Exception as e:
                raise RuntimeError("distance mode requires scipy.ndimage.distance_transform_edt") from e

            # binary solid from level-set (classical gyroid surface at 'level')
            # first create binary mask of solid region, the surface of interest is at the intersection of the two regions
            binary = (term > 0)

            # distance_transform_edt supports a 'sampling' parameter for anisotropic voxels
            # second, compute in the solid part, the distance of every voxel to the nearest zero (empty part)
            if mode == "distance":
                dist_out = distance_transform_edt(~binary, sampling=spacing)
                # third, do the same, but inverting the regions
                dist_in = distance_transform_edt(binary, sampling=spacing)
            else:
                logger.warning("Using FAST distance transform does not work for anisotropic voxels.")
                # for a faster but less accurate approximation, compute the distance in the binary mask without inverting it
                dist_out = distance_transform_cdt(~binary, metric="taxicab")
                # third, do the same, but inverting the regions
                dist_in = distance_transform_cdt(binary, metric="taxicab")

            # distance: now the matrx shows the distance to the surface
            dist = dist_out + dist_in

            # actual distance to the surface need to be half the total distance
            half_t = self.thickness / 2.0

            #crate a mask of the voxel the are below the max distance
            mask = dist < half_t

            # final field: positive inside the desired wall band, zero outside
            self.v = np.zeros_like(dist) - 1
            self.v[mask] = dist[mask]
            return self.v

        raise ValueError("mode must be one of: 'abs', 'signed', 'distance', 'distance_fast'")

    # =====================================================================
    # 5) save
    # =====================================================================
    def save(self, outfile: str) -> None:
        """
        ============================================================================
        5) SAVE
        Persists gyroid parameters and the computed field to disk using the
        package I/O helper.
        ============================================================================

        PARAMETERS
        ----------
        outfile : str
            Path to the output .npz file.

        RETURNS
        -------
        None
        """
        if self.v is None:
            raise RuntimeError("Gyroid field has not been computed yet (call compute_field).")

        io_ops.save_gyroid_matrices(
            outfile,
            Xres = self.x,
            Yres = self.y,
            Zres = self.z,
            Xperiod=self.px,
            Yperiod=self.py,
            Zperiod=self.pz,
            thickness=self.thickness,
            gyroid_field=self.v,
        )

    # =====================================================================
    # 6) load
    # =====================================================================
    @classmethod
    def load(cls, infile: str) -> "GyroidModel":
        """
        ============================================================================
        6) LOAD
        Loads saved gyroid matrices and returns a GyroidModel instance.
        ============================================================================

        PARAMETERS
        ----------
        infile : str
            Path to the input .npz file.

        RETURNS
        -------
        model : GyroidModel
            A GyroidModel populated with the saved coordinates, parameters,
            and field. Mesh data (verts/faces) is not stored on disk and is
            reset to None.

        EXAMPLE
        -------
        >>> model = GyroidModel.load("gyroid_data.npz")
        """
        x, y, z, px, py, pz, t, v = io_ops.load_gyroid_matrices(infile)
        obj = cls.__new__(cls)
        # fill only what is stored; coordinates are unknown from the saved file
        obj.x = x
        obj.y = y
        obj.z = z
        obj.px, obj.py, obj.pz = px, py, pz
        obj.thickness = t
        obj.v = v
        obj.verts = None
        obj.faces = None
        return obj

    # =====================================================================
    # 7) generate_mesh
    # =====================================================================
    def generate_mesh(
        self,
        iso_level: float = 0.0,
        algo_step_size: int = 3,
        pad_width: int = 5,
        pad_val: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        ============================================================================
        7) GENERATE_MESH
        Generates a triangular surface mesh from the scalar field using the
        mesh_tools helper.
        ============================================================================

        PARAMETERS
        ----------
        iso_level : float, optional
            Isosurface level (default = 0.0).
        algo_step_size : int, optional
            Marching cubes step size (default = 3).
        pad_width : int, optional
            Number of voxels to pad on each face of the volume (default = 5).
        pad_val : float, optional
            Constant padding value (default = 0.0).

        RETURNS
        -------
        verts, faces : (np.ndarray, np.ndarray)
            The generated vertices and faces (also stored on self).
        """
        if self.v is None:
            logger.error("Gyroid field has not been computed yet. Call compute_field() before generate_mesh().")
            return None, None

        self.verts, self.faces = mesh_tools.mesh_from_matrix(
            matrix=self.v,
            iso_level=iso_level,
            algo_step_size=algo_step_size,
            x=self.x,
            y=self.y,
            z=self.z,
            pad_width=pad_width,
            pad_val=pad_val,
        )

        logger.info(f"Generated mesh with {len(self.faces)} faces")
        return self.verts, self.faces

    # =====================================================================
    # 8) simplify_mesh
    # =====================================================================
    def simplify_mesh(self, target_faces: int = 10000, mode: str = "normal"):
        """
        ============================================================================
        8) SIMPLIFY_MESH
        Simplifies and cleans the current mesh, returning (verts, faces).
        Uses the mesh_tools simplification and connected-component filtering
        helpers.
        ============================================================================

        PARAMETERS
        ----------
        target_faces : int, optional
            Target number of faces to keep (default = 10000).
        mode : str, optional
            "normal" (default) or "fast".

        RETURNS
        -------
        verts, faces : (np.ndarray, np.ndarray)
            The simplified vertices and faces (also stored on self).
        """
        if self.verts is None or self.faces is None:
            logger.error("Mesh has not been generated yet.")
            return

        if mode == "fast":
            self.faces, self.verts = mesh_tools.simplify_mesh(self.faces, self.verts, target=target_faces, mode="fast")
        else:
            self.faces, self.verts = mesh_tools.simplify_mesh(self.faces, self.verts, target=target_faces, mode="normal")

        # keep the largest connected component and discard stray pieces
        self.verts, self.faces = mesh_tools.keep_largest_connected_component(self.verts, self.faces)

        logger.info(f"Mesh simplified to {len(self.faces)} faces")
        return

    # =====================================================================
    # 9) export_stl
    # =====================================================================
    def export_stl(self, filepath: str) -> None:
        """
        ============================================================================
        9) EXPORT_STL
        Exports the current mesh as an STL file.
        ============================================================================

        PARAMETERS
        ----------
        filepath : str
            Output path (without extension); ".stl" is appended.

        RETURNS
        -------
        None
        """
        if self.verts is None or self.faces is None:
            logger.error("Mesh has not been generated yet.")
            return

        mesh_tools.export_as_STL(self.verts, self.faces, filepath+'.stl')
        logger.info(f"STL exported to: {filepath}.stl")

    # =====================================================================
    # 10) save_mesh_preview
    # =====================================================================
    def save_mesh_preview(self, html_path: str, show_normal_colorscale: bool = True) -> None:
        """
        ============================================================================
        10) SAVE_MESH_PREVIEW
        Saves an interactive HTML preview of the mesh (via viz helper).
        ============================================================================

        PARAMETERS
        ----------
        html_path : str
            Output HTML file path (without extension).
        show_normal_colorscale : bool, optional
            If True (default), colors faces based on normal vectors.

        RETURNS
        -------
        None
        """
        if self.verts is None or self.faces is None:
            logger.error("Mesh has not been generated yet.")
            return

        viz.save_mesh_as_html(self.faces, self.verts, html_path, show_normal_colorscale=show_normal_colorscale)

    # =====================================================================
    # 11) check_mesh_quality
    # =====================================================================
    def check_mesh_quality(self) -> bool:
        """
        ============================================================================
        11) CHECK_MESH_QUALITY
        Checks mesh validity and returns a boolean indicating if the mesh is
        valid.
        ============================================================================

        PARAMETERS
        ----------
        None

        RETURNS
        -------
        is_valid : bool
            True if the mesh is watertight, winding-consistent, and not
            self-intersecting.
        """
        if self.verts is None or self.faces is None:
            raise RuntimeError("Mesh has not been generated yet.")

        #areas = mesh_tools.calculate_triangle_areas(self.verts, self.faces)
        #viz.plot_histogram(areas)
        info = mesh_tools.check_mesh_validity(self.verts, self.faces)
        if info["watertight"] and info["winding_consistent"] and not info["self_intersecting"]:
            validty = True
        else:
            validty = False
        return validty


    # =====================================================================
    # 12) keep_largest_connected_component
    # =====================================================================
    def keep_largest_connected_component(self) :
        """
        ============================================================================
        12) KEEP_LARGEST_CONNECTED_COMPONENT
        Convenience wrapper to the mesh_tools function.
        ============================================================================

        PARAMETERS
        ----------
        None

        RETURNS
        -------
        None
        """
        self.verts, self.faces = mesh_tools.keep_largest_connected_component(self.verts, self.faces)


    # =====================================================================
    # 13) add_baseplates
    # =====================================================================
    def add_baseplates(
            self,
            thickness: float = 5.0,
        ) -> None:
        """
        ============================================================================
        13) ADD_BASEPLATES
        Adds solid baseplates on the two ends of the z-axis with given physical
        thickness (same units as self.z). The method preserves the 3D shape of
        self.v and sets voxels inside the baseplate regions to 1.
        ============================================================================

        PARAMETERS
        ----------
        thickness : float, optional
            Baseplate thickness in the same units as self.z (default = 5.0).

        RETURNS
        -------
        None
        """
        if self.v is None:
            raise RuntimeError("Field not computed: call compute_field() before add_baseplates().")
        if self.z is None:
            raise RuntimeError("Grid coordinates missing: self.z is required to compute baseplate thickness in z.")

        # extract 1D z-coordinate along the third axis (assumes indexing 'ij' meshgrid)
        z_line = np.asarray(self.z[0, 0, :])
        if z_line.ndim != 1:
            raise RuntimeError("Unexpected z-grid shape; expected 1D slice along z axis at [0,0,:].")

        # count how many z-slices lie below the requested thickness
        n = np.abs(self.z[0,0,1] - self.z[0,0,0])  # distance of one slice
        N = int(thickness/n)                       # number of slices to fill

        # clamp to valid range
        nz = self.v.shape[2]
        if N <= 0:
            logger.info("Requested baseplate thickness is zero or smaller than grid spacing; no baseplates added.")
            return
        if N >= nz:
            logger.warning("Requested baseplate thickness >= entire z-size; filling whole volume.")
            N = nz

        # set the bottom and top n slices to solid (use in-place assignment to preserve dtype/shape)
        self.v[:, :, 0:N] = 1
        self.v[:, :, -N:] = 1

        logger.info(f"Added baseplates of thickness {thickness} units ({N} z-slices).")

    # =====================================================================
    # 14) fix_mesh
    # =====================================================================
    def fix_mesh(self):
        """
        ============================================================================
        14) FIX_MESH
        Convenience wrapper to the mesh_tools fix_mesh function.
        ============================================================================

        PARAMETERS
        ----------
        None

        RETURNS
        -------
        None
        """
        if self.verts is None or self.faces is None:
            raise RuntimeError("Mesh has not been generated yet.")

        self.verts, self.faces = mesh_tools.fix_mesh(self.verts, self.faces)

    # =====================================================================
    # 15) smooth_mesh
    # =====================================================================
    def smooth_mesh(self, smoothing_factor: float = 0.5):
        """
        ============================================================================
        15) SMOOTH_MESH
        Convenience wrapper to the mesh_tools.smooth_mesh function.
        ============================================================================

        PARAMETERS
        ----------
        smoothing_factor : float, optional
            Taubin smoothing lambda parameter (default = 0.5).

        RETURNS
        -------
        None
        """
        if self.verts is None or self.faces is None:
            raise RuntimeError("Mesh has not been generated yet.")

        self.verts, self.faces = mesh_tools.smooth_mesh(self.verts, self.faces, smoothing_factor=smoothing_factor)


# =====================================================================
# 16) create_a_gyroid
# =====================================================================
def create_a_gyroid(x:np.ndarray,
                    y:np.ndarray,
                    z:np.ndarray,
                    px:np.ndarray,
                    py:np.ndarray,
                    pz:np.ndarray,
                    t:np.ndarray,
                    save_path: str,
                    baseplate_thickness: float = 0.0,
                    step_size:int=2,
                    simplification_factor=0.9,
                    field_mode:str = "distance"):
    """
    ============================================================================
    16) CREATE_A_GYROID
    Convenience function to create a gyroid model, compute the field,
    generate and simplify the mesh, and save results.
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
        Thickness of the baseplates to add (same units as z). 0 = none (default).
    step_size : int, optional
        Marching cubes step size (higher = faster but less detailed mesh, default = 2).
    simplification_factor : float, optional
        Target fraction of faces to keep during simplification (0.5 = keep
        50% of faces), or target number of faces if >1 (e.g. 10000). Default = 0.9.
    field_mode : str, optional
        Field computation mode passed to GyroidModel.compute_field
        (default = "distance").

    RETURNS
    -------
    success : bool
        True if a valid mesh was generated and exported, False otherwise.
    """
    #make the gyroid model with distance field
    model_dist = GyroidModel(x, y, z, px, py, pz, t)
    model_dist.compute_field(mode = field_mode)
    if baseplate_thickness > 0.0:
        model_dist.add_baseplates(thickness=baseplate_thickness)
    #model_dist.save(save_path + ".npz")

    #generate mesh
    model_dist.generate_mesh(algo_step_size=step_size)
    model_dist.smooth_mesh(smoothing_factor= 0.9)

    model_dist.simplify_mesh(target_faces = simplification_factor, mode="fast")
    model_dist.smooth_mesh(smoothing_factor= 0.6)
    model_dist.fix_mesh()
    is_valid = model_dist.check_mesh_quality()
    #save preview and stl
    model_dist.save_mesh_preview(save_path)
    if not is_valid:
        logger.warning("Generated mesh is not valid. Will ignore this one.")
        return False  # Signal failure to caller
    model_dist.export_stl(save_path)
    return True  # Signal success to caller

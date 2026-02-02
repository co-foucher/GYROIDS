import numpy as np
from typing import Optional, Tuple, Union

# Use relative imports so this module works when used as a package
from . import io_ops, mesh_tools, viz
from .logger import logger


class GyroidModel:
    """
    Represents a gyroid scalar field defined on a 3D grid and provides helpers
    to compute the field, generate/simplify a mesh, export results and run
    simple checks/visualizations.

    Expected inputs:
    - x, y, z: numpy arrays of identical shape describing coordinates for the field.
    - px, py, pz: periods. Each may be a scalar (most common) or an array with the
      same shape as x/y/z (for per-voxel period variations).
    - thickness: scalar or array (shape identical to x/y/z) controlling the isosurface threshold.

    usage:
        model = GyroidModel(x, y, z, px, py, pz, thickness)
        field = model.compute_field() 

    classmethods:
    - load: load saved gyroid parameters and field from disk. usgae: GyroidModel.load(infile)
    """

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

    def _validate_inputs(self):
        """
        Validate shapes/types of inputs.
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

    def compute_field(self,
                      mode: str = "abs",
                      spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)) -> np.ndarray:
        """
        Compute the gyroid scalar field.

        mode:
          - "abs" (default): original behavior -> v = thickness - |term|
            (useful for value-space wall thresholding; thickness here is in term units)
          - "signed": standard level-set -> v = term - level  (signed field)
          - "distance": produce a signed-distance-derived thickness field:
                1) binary = term > level
                2) compute signed distance (uses spacing)
                3) if physical_thickness provided (scalar or array matching grid), v = physical_thickness/2 - |signed_dist|
                   (positive inside the desired wall band)

        spacing: voxel spacing used when computing distance transform (only for mode="distance").
        physical_thickness: desired wall thickness in spatial units (only used for "distance" mode).
                            May be a scalar or an ndarray with the same shape as x/y/z.
        """
        term = (
            np.sin((2 * np.pi / self.px) * self.x) * np.cos((2 * np.pi / self.py) * self.y)
            + np.sin((2 * np.pi / self.py) * self.y) * np.cos((2 * np.pi / self.pz) * self.z)
            + np.sin((2 * np.pi / self.pz) * self.z) * np.cos((2 * np.pi / self.px) * self.x)
        )

        if mode == "abs":
            # original behaviour: thickness interpreted in term-value units (supports scalar or per-voxel thickness)
            self.v = self.thickness - np.abs(term)
            return self.v

        if mode == "signed":
            # signed level-set relative to provided level (C)
            self.v = term - self.thickness
            return self.v

        if mode == "distance":
            # requires scipy
            try:
                from scipy.ndimage import distance_transform_edt
            except Exception as e:
                raise RuntimeError("distance mode requires scipy.ndimage.distance_transform_edt") from e

            # binary solid from level-set (classical gyroid surface at 'level')
            # first create binary mask of solid region, the surface of interest is at the intersection of the two regions
            binary = (term > 0)

            # distance_transform_edt supports a 'sampling' parameter for anisotropic voxels
            # second, compute in the solid part, the distance of every voxel to the nearest zero (empty part)
            dist_out = distance_transform_edt(~binary, sampling=spacing)
            # third, do the same, but inverting the regions
            dist_in = distance_transform_edt(binary, sampling=spacing)

            # distance: now the matrx shows the distance to the surface
            dist = dist_out + dist_in

            # actual distance to the surface need to be half the total distance
            half_t = self.thickness / 2.0

            #crate a mask of the voxel the are below the max distance
            mask = dist < half_t

            # final field: positive inside the desired wall band, zero outside
            self.v = np.zeros_like(dist)
            self.v[mask] = dist[mask]
            return self.v

        raise ValueError("mode must be one of: 'abs', 'signed', 'distance'")

    def save(self, outfile: str) -> None:
        """
        Persist gyroid parameters and the computed field to disk using the package I/O helper.
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

    @classmethod
    def load(cls, infile: str) -> "GyroidModel":
        """
        Load saved gyroid matrices and return a GyroidModel instance.
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

    def generate_mesh(
        self,
        iso_level: float = 0.0,
        spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        algo_step_size: int = 3,
        pad_width: int = 5,
        pad_val: float = 0.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate a triangular surface mesh from the scalar field using the mesh_tools helper.
        Returns (verts, faces).
        """
        if self.v is None:
            raise RuntimeError("Gyroid field has not been computed yet.")

        self.verts, self.faces = mesh_tools.mesh_from_matrix(
            matrix=self.v,
            iso_level=iso_level,
            spacing=spacing,
            algo_step_size=algo_step_size,
            x=0.0,
            y=0.0,
            z=0.0,
            pad_width=pad_width,
            pad_val=pad_val,
        )

        logger.info(f"Generated mesh with {len(self.faces)} faces")
        return self.verts, self.faces

    def simplify_mesh(self, target_faces: int = 10000) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simplify and clean the current mesh, returning (verts, faces).
        This uses the mesh_tools simplification and connected-component filtering helpers.
        """
        if self.verts is None or self.faces is None:
            raise RuntimeError("Mesh has not been generated yet.")

        self.faces, self.verts = mesh_tools.simplify_mesh(self.faces, self.verts, target=target_faces)

        # keep the largest connected component and discard stray pieces
        self.verts, self.faces = mesh_tools.keep_largest_connected_component(self.verts, self.faces)

        logger.info(f"Mesh simplified to {len(self.faces)} faces")
        return self.verts, self.faces

    def export_stl(self, filepath: str) -> None:
        """
        Export the current mesh as an STL file.
        """
        if self.verts is None or self.faces is None:
            raise RuntimeError("Mesh has not been generated yet.")

        mesh_tools.export_as_STL(self.verts, self.faces, filepath)
        logger.info(f"STL exported to: {filepath}")

    def save_mesh_preview(self, html_path: str) -> None:
        """
        Save an interactive HTML preview of the mesh (via viz helper).
        """
        if self.verts is None or self.faces is None:
            raise RuntimeError("Mesh has not been generated yet.")

        viz.save_mesh_as_html(self.faces, self.verts, html_path)

    def check_mesh_quality(self) -> np.ndarray:
        """
        Calculate triangle areas, plot a histogram and run mesh validity checks.
        Returns the computed areas array.
        """
        if self.verts is None or self.faces is None:
            raise RuntimeError("Mesh has not been generated yet.")

        areas = mesh_tools.calculate_triangle_areas(self.verts, self.faces)
        viz.plot_histogram(areas)
        mesh_tools.check_mesh_validity(self.verts, self.faces)
        return areas


    def keep_largest_connected_component(
        verts: np.ndarray,
        faces: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convenience wrapper to the mesh_tools function.
        """
        return mesh_tools.keep_largest_connected_component(verts, faces)
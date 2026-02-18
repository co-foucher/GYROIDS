# **GYROIDS**
  This is a small library to support the development of gyroid-based structures. The general idea to use it follows this structure:

<img width="1894" height="921" alt="image" src="https://github.com/user-attachments/assets/c65cae60-cd07-47e0-a794-d1a3a486b6e0" />

# **REQUIREMENTS**
  !!!! requires Python 3.10 !!!!

# **INSTALLATION**
  - You could copy the whole environment directly from the yml file. Or use pip to install only this library and its dependency. 
  
  - It is better to first create a python 3.10 venv and then use pip install git+https://github.com/co-foucher/GYROIDS.git

```powershell
      conda create -n nameofenv python=3.10
      pip install git+https://github.com/co-foucher/GYROIDS.git
```
  - For changes: update the toml file and then use pip install git+https://github.com/co-foucher/GYROIDS.git



# Module Organization

The library is organized into the following modules:

- **gyroid.py**: Main GyroidModel class and convenience functions
- **mesh_tools.py**: Mesh processing functions (simplification, smoothing, fixing, validation, export)
- **io_ops.py**: Input/output operations (STL loading/saving, .npz archives)
- **viz.py**: Visualization tools (HTML previews, histograms, 2D matrix views)
- **abaqus_tools.py**: ABAQUS simulation integration
- **TET_mesh_tools.py**: Tetrahedral mesh operations
- **logger.py**: Logging configuration
- **config.py**: Configuration constants

# Examples in Notebooks
- **Gyroids_STL.ipynb**: how to parametrize and create a simple gyroid
- **Gyroids_STL_class.ipynb**: how to use the gyroid class to generate gyroids faster and safer
- **STL_to_STEP.ipynb**: how to use OCC to transform a STL to a STEP
- **STL_to_inp.ipynb**: how to use fTetWild to transform a STL to an ABAQUS input file (inp)


# FEATURES
### Complex Gyroid Generation
- Three field computation modes: 'abs', 'signed', and 'distance'
- Support for variable periods and thickness (scalar or per-voxel arrays)
- 2D view of matrix defining the gyroid
- Save/load gyroid field data as compressed .npz archives
- Add baseplates to gyroid structures

### Surface Mesh Processing (STL)
- Create a mesh from matrix using marching cubes algorithm
- Fast mesh decimation with PyVista/VTK
- Quadric mesh simplification (Open3D) 
- Mesh smoothing (Humphrey filter)
- Mesh fixing and repair (Trimesh, PyMeshFix)
- Triangle area computation (NumPy)  
- Extract the largest connected component 
- Check mesh validity (watertight, manifold, self-intersecting)
- Mesh preview as interactive HTML (Plotly)
- Histogram visualization of triangle sizes
- Export as STL

### Tetrahedral Meshing
- ftetwild: to go from STL to tetrahedral mesh, check out ftetwild : https://github.com/wildmeshing/fTetWild

### ABAQUS Integration
- Create ABAQUS simulations from meshes
- Run frequency and full DSS simulations
- Automated simulation file generation

### Input / Output
- Robust STL loading and export (Open3D, numpy-stl)
- Save/load gyroid parameters and fields (.npz format)
- Configurable logging for all operations

### Utilities
- Set log levels (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- Module reload functionality for development
- Help system

# GyroidModel (class: src/gyroid_utils/gyroid.py)
A high-level helper class to create a gyroid scalar field on a 3D grid, convert it into a surface mesh, simplify/export the mesh and run quick checks/visualizations.

Quick summary
- Location: src/gyroid_utils/gyroid.py
- Main purpose: generate a gyroid (level-set) field, produce a triangular mesh, and export/inspect it.
- Typical workflow:
  1. Instantiate GyroidModel with coordinate grids and parameters.
  2. Call compute_field(...) to build the scalar field.
  3. Call generate_mesh(...) to extract an isosurface mesh (default iso_level = 0).
  4. Optionally simplify, smooth, fix, export to STL/STEP, preview or check mesh quality.

## Constructor
- **GyroidModel(x, y, z, px, py, pz, thickness)**
  - x, y, z: numpy ndarrays describing the sampling grid (must have identical shape).
  - px, py, pz: period parameters (scalar or ndarray matching x.shape).
  - thickness: scalar or ndarray (same shape as x) used to control wall thickness.

## Main Methods

### compute_field(mode, spacing)
Builds the internal scalar field self.v. 

**Modes:**
- **mode="abs"** (default): `v = thickness - |term|`
  - Fast, threshold in function-value space. Good for quick "wall" masks.
  
- **mode="signed"**: `v = term - level`
  - Classical signed level-set (use level=0 for canonical gyroid).
  
- **mode="distance"**: builds a distance-based field using scipy.ndimage.distance_transform_edt
  - Produces a signed-distance field based on spatial thickness.
  - spacing parameter controls voxel sizes for the distance transform (tuple: (x_spacing, y_spacing, z_spacing)).
  - thickness is interpreted as physical wall thickness in spatial units.


### generate_mesh(iso_level, algo_step_size, pad_width, pad_val)
Produces (verts, faces) from self.v using marching-cubes (mesh_tools.mesh_from_matrix).

**Parameters:**
- iso_level: float - isosurface level (default: 0.0)
- algo_step_size: int - marching cubes step size (default: 3)
- pad_width: int - padding width (default: 5)
- pad_val: float - padding value (default: 0.0)

### simplify_mesh(target_faces, mode)
Simplify mesh and keep largest connected component. Modes: 'fast' (PyVista) or 'slow' (Open3D).

### smooth_mesh(smoothing_factor)
Apply Humphrey smoothing filter to mesh.

### fix_mesh()
Fix mesh issues (non-manifold edges, holes, etc.) using Trimesh repair.

### add_baseplates(thickness)
Add solid baseplates on bottom and top (z-axis).

### export_stl(filepath)
Save mesh as STL.

### save_mesh_preview(html_path, show_normal_colorscale)
Save interactive HTML preview using Plotly.

### check_mesh_quality()
Compute triangle areas and run validity checks.

### keep_largest_connected_component()
Keep only the largest connected component, remove others.

### save(outfile)
Persist gyroid parameters and field to .npz archive.

### load(infile) [classmethod]
Load saved gyroid parameters and field from disk. Returns GyroidModel instance.

## Convenience Function: create_a_gyroid()

**create_a_gyroid(x, y, z, px, py, pz, t, save_path, baseplate_thickness, step_size, simplification_factor)**

All-in-one function to create a gyroid model, compute the field, generate and simplify the mesh, and save results.

**Parameters:**
- x, y, z: coordinate grids (3D arrays of identical shape)
- px, py, pz: periods (scalars or arrays matching x/y/z shape)
- t: thickness parameter (scalar or array matching x/y/z shape)
- save_path: base path for saving the .npz field and .stl mesh (without extension)
- baseplate_thickness: thickness of the baseplates to add in spatial units (default: 0.0)
- step_size: marching cubes step size - higher = faster but less detailed mesh (default: 2)
- simplification_factor: target fraction of faces to keep (0.5 = keep 50% of faces) or target number of faces if >1 (e.g. 10000) (default: 0.9)

## Minimal Example
```python
import numpy as np
from gyroid_utils.gyroid import GyroidModel

# build grid
x, y, z = np.meshgrid(np.linspace(0,1,64),
                      np.linspace(0,1,64),
                      np.linspace(0,1,64), indexing='ij')

model = GyroidModel(x, y, z, px=1.0, py=1.0, pz=1.0, thickness=0.2)
# distance-based wall of physical thickness 0.5 (spatial units), requires scipy
model.compute_field(mode='distance', spacing=(1.0,1.0,1.0))
model.add_baseplates(thickness=5.0)  # optional: add baseplates
verts, faces = model.generate_mesh(iso_level=0.0)
model.smooth_mesh(smoothing_factor=0.5)  # optional: smooth mesh
model.simplify_mesh(target_faces=10000, mode='fast')
model.fix_mesh()  # optional: fix mesh issues
model.export_stl("gyroid")
model.save_mesh_preview("gyroid_preview")
model.save("gyroid_data.npz")
```

## Using the Convenience Function
```python
import numpy as np
from gyroid_utils.gyroid import create_a_gyroid

# build grid
x, y, z = np.meshgrid(np.linspace(0,10,128),
                      np.linspace(0,10,128),
                      np.linspace(0,20,256), indexing='ij')

create_a_gyroid(
    x, y, z,
    px=2.0, py=2.0, pz=2.0,
    t=1.0,
    save_path="my_gyroid",
    baseplate_thickness=2.0,
    step_size=2,
    simplification_factor=0.8  # keep 80% of faces
)
```

# Logging

Control logging verbosity:

```python
import gyroid_utils
gyroid_utils.set_log_level("DEBUG")  # or "INFO", "WARNING", "ERROR", "CRITICAL"
```

# Known Bugs
Coordinates in the STL mesh do not match exactly the definition in the matrix. This is due to the marching cube algorithm, resulting in structure about a pixel larger in every dimension.
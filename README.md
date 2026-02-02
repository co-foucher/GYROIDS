# **GYROIDS**
  This is a small library to support the development of gyroid-based structures. The general idea to use it follows this structure:

<img width="1894" height="921" alt="image" src="https://github.com/user-attachments/assets/c65cae60-cd07-47e0-a794-d1a3a486b6e0" />

# **REQUIREMENTS**
  !!!! requires Python 3.10 !!!!
  You'll need conda to install OCC, because it is in Python 3.10

# **INSTALLATION**
  - You could copy the whole environment directly from the yml file. Or use pip to install only this library and *some* dependency. 
  
  - It is better to first create a python 3.10 venv and then use pip install git+https://github.com/co-foucher/GYROIDS.git
  and finally add OCC through: conda install conda-forge::pythonocc-core

```powershell
      conda create -n nameofenv python=3.10
      pip install git+https://github.com/co-foucher/GYROIDS.git
      conda install conda-forge::pythonocc-core 
  ```
  - For changes: update the toml file and then use pip install git+https://github.com/co-foucher/GYROIDS.git


# FEATURES
### Complex Gyroid Generation
- 2D view of matrix defining the gyroid
- example of gyroid definition

### Surface Mesh Processing (STL)
- Create a mesh from matrix
- Quadric mesh simplification (Open3D) 
- Triangle area computation (NumPy)  
- Mesh preview as interactive HTML (Plotly)
- Extract the largest connected component 
- Histogram visualization of triangle sizes
- check_mesh_validity
- export as an STL 

### OpenCascade Tools to generate STEP from STL
- Build planar faces from triangle data  
- Stitch faces into closed shells  
- Validate and simplify OCC shapes  
- Export shapes to STEP files  
- Check if a shell is watertight/closed

### ftetwild
-to go from stl to tetrahedral mesh, check out ftetwild : https://github.com/wildmeshing/fTetWild

### Input / Output
- Robust STL export and loading 
- STEP export (slow)
- Configurable logging for all operations

# GyroidModel (class: src/gyroid_utils/gyroid.py)
A high-level helper class to create a gyroid scalar field on a 3D grid, convert it into a surface mesh, simplify/export the mesh and run quick checks/visualizations.

Quick summary
- Location: src/gyroid_utils/gyroid.py
- Main purpose: generate a gyroid (level-set) field, produce a triangular mesh, and export/inspect it.
- Typical workflow:
  1. Instantiate GyroidModel with coordinate grids and parameters.
  2. Call compute_field(...) to build the scalar field.
  3. Call generate_mesh(...) to extract an isosurface mesh (default iso_level = 0).
  4. Optionally simplify, export to STL/STEP, preview or check mesh quality.

Constructor
- GyroidModel(x, y, z, px, py, pz, thickness)
  - x, y, z: numpy ndarrays describing the sampling grid (must have identical shape).
  - px, py, pz: period parameters (scalar or ndarray matching x.shape).
  - thickness: scalar or ndarray (same shape as x) used to control wall thickness in some modes.

main helper: ompute_field(...)
- Builds the internal scalar field self.v. Modes:
  - mode="abs" (default): v = thickness - |term|
    - Fast, threshold in function-value space. Good for quick "wall" masks.
  - mode="signed": v = term - level
    - Classical signed level-set (use level=0 for canonical gyroid).
  - mode="distance": builds a distance-based field using scipy.ndimage.distance_transform_edt
    - Produces a signed-distance field (if physical_thickness is None) or a band-field
      based on a spatial thickness (if physical_thickness provided).
    - spacing parameter controls voxel sizes for the distance transform.
    - physical_thickness may be scalar or an ndarray matching the grid.

Other helpers
- generate_mesh(...): Produces (verts, faces) from self.v using marching-cubes (mesh_tools.mesh_from_matrix).
- simplify_mesh(target_faces): simplify and retain largest connected component.
- export_stl(filepath): save current mesh as STL.
- save_mesh_preview(html_path): save interactive HTML preview.
- check_mesh_quality(): compute triangle areas, plot histogram, run validity checks.


Minimal example
```python
import numpy as np
from gyroid_utils.gyroid import GyroidModel

# build grid
x, y, z = np.meshgrid(np.linspace(0,1,64),
                      np.linspace(0,1,64),
                      np.linspace(0,1,64), indexing='ij')

model = GyroidModel(x, y, z, px=1.0, py=1.0, pz=1.0, thickness=0.2)
# distance-based wall of physical thickness 0.5 (voxel units), requires scipy
model.compute_field(mode='distance', spacing=(1.0,1.0,1.0), thickness=0.5)
verts, faces = model.generate_mesh(iso_level=0.0)
model.simplify_mesh(target_faces=10000)
model.export_stl("gyroid.stl")
```
# Examples in notebooks
- Gyroids_STL.ipynb : how to parametrize and create a simple gyroid
- Gyroids_STL_class.ipynb : how to use the gyroid class to generate gyroids faster and safer
- STL_to_STEP.ipynb : how to use OCC to transform a STL to a STEP
- STL_to_inp.ipynb : how to use fTetWild to transform a STL to an ABAQUS input file (inp)

# known bugs
Coordinates in the stl mesh do not match exactly the definition in the matrix. his is due to the marching cube algorithm, resulting in strcute about a pixel larger in every dimension.




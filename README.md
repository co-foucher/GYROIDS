# **GYROIDS UTILS**

  This is a small library to support the development of TPMS structures. It is developed around three use cases, and its structure is shown below

 <img width="788" height="422" alt="image" src="https://github.com/user-attachments/assets/b63c8425-c57e-4a16-876f-61e976fb5344" />

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

# Known Bugs
Coordinates in the STL mesh do not match exactly the definition in the matrix. This is due to the marching cube algorithm, resulting in structure about a pixel larger in every dimension.

# Module Organization
The library is organized around three main use cases:

## TPMS structure generation
This is used to generate TPMS structures (especially gyroids) using the general workflow below:
<img width="1894" height="921" alt="image" src="https://github.com/user-attachments/assets/c65cae60-cd07-47e0-a794-d1a3a486b6e0" />
note that it was originaly designed for creating gyroid, but not limited to them.

Scripts related to this use case:
- **gyroid.py**: Main GyroidModel class and convenience functions
- **mesh_tools.py**: Mesh processing functions (simplification, smoothing, fixing, validation, export)
- **io_ops.py**: Input/output operations (STL loading/saving, .npz archives)
- **viz.py**: Visualization tools (HTML previews, histograms, 2D matrix views)

Example notebooks for this use case:
- **Gyroids_STL.ipynb**
- **Gyroids_STL_class.ipynb**

## Simulation of STL file
This is used to be able to create simulations of the generated structures. Morespecificaly to create tetrahedral mesh adapted to finite element modeling using FtetWild, manipulate them, create ABAQUS input files, and run them in batches.

Scripts related to this use case:
- **abaqus_tools.py**: ABAQUS simulation integration
- **TET_mesh_tools.py**: Tetrahedral mesh operations

Example notebooks for this use case:
- **full simulation workflow.ipynb**: end-to-end simulation preparation workflow
- **STL_to_inp_ftetwild.ipynb**: how to use fTetWild to transform a STL to an ABAQUS input file (inp)

## Analysis of CT scans
This is used to help analyse CT scan of structures.

Scripts related to this use case:
- **CT_scans.py**: CT data readers and preprocessing helpers
- **CT_visualization_window.py**: CT visualization tooling

Example notebooks for this use case:
- **CT_scan_processing.ipynb**: CT-to-mesh pipeline and interactive mesh coloring (including curvature mode)

## library tools and configuration
Other scripts exist for configuring this library and some usefull functions
- **logger.py**: Logging configuration


# FEATURES

## TPMS Structure Generation
- Three field computation modes: `'abs'`, `'signed'`, and `'distance'` for flexible wall definition
- Support for variable periods and thickness (scalar or per-voxel arrays) of gyroids
- Optional baseplates for structural support

## Surface Mesh Processing
- Marching cubes algorithm for isosurface extraction
- Multiple mesh simplification strategies: fast (PyVista) or high-quality (Open3D)
- Mesh smoothing with Humphrey filter
- Automatic mesh repair (non-manifold edges, hole filling)
- Comprehensive mesh validation (watertight, manifold, self-intersections)
- Interactive HTML previews with Plotly, with different color scheme: constant, random, normal, curvature.
- Triangle area analysis and visualization
- Robust STL import/export (Open3D and numpy-stl backends)

## Tetrahedral Meshing
- Seamless integration with [fTetWild](https://github.com/wildmeshing/fTetWild) for high-quality mesh generation
- Tetrahedral mesh manipulation and refinement tools

## ABAQUS Simulation
- Automated frequency analysis simulations
- Full DSS (Dynamic Substructuring) simulations
- Batch simulation file generation and management

## CT Scan Analysis
- CT data reading and preprocessing
- Interactive visualization window for CT scans
- Mesh coloring and analysis (including curvature visualization)

## Utilities
- Compressed field data storage (.npz format)
- Configurable logging (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- Module reload functionality for interactive development


# Quick Start: Creating a Gyroid
The `GyroidModel` class is the main entry point for gyroid generation. The typical workflow is:

1. Create a coordinate grid
2. Instantiate `GyroidModel` with your grid and parameters
3. Compute the scalar field
4. Generate and simplify mesh
5. Export as STL

### Simple Example

```python
import numpy as np
from gyroid_utils.gyroid import GyroidModel

# Create a 64×64×64 grid
x, y, z = np.meshgrid(np.linspace(0,1,64),
                      np.linspace(0,1,64),
                      np.linspace(0,1,64), indexing='ij')

# Initialize model
model = GyroidModel(x, y, z, px=1.0, py=1.0, pz=1.0, thickness=0.2)

# Build scalar field (choose 'abs', 'signed', or 'distance')
model.compute_field(mode='distance', spacing=(1.0, 1.0, 1.0))

# Generate mesh from isosurface
verts, faces = model.generate_mesh()

# Simplify, smooth, and repair
model.simplify_mesh(target_faces=10000, mode='fast')
model.smooth_mesh(smoothing_factor=0.5)
model.fix_mesh()

# Export
model.export_stl("my_gyroid.stl")
model.save("gyroid_data.npz")  # Save field for later
```

### All-in-One Function

For a quicker workflow, use `create_a_gyroid()`:

```python
import numpy as np
from gyroid_utils.gyroid import create_a_gyroid

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
    simplification_factor=0.8
)
```

For more detailed API documentation and parameters, see [gyroid.py](src/gyroid_utils/gyroid.py) or check out the example notebooks.


# Logging
Control logging verbosity:

```python
import gyroid_utils
gyroid_utils.set_log_level("DEBUG")  # or "INFO", "WARNING", "ERROR", "CRITICAL"
```



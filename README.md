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

### Input / Output
- Robust STL export and loading 
- STEP export  
- Configurable logging for all operations 


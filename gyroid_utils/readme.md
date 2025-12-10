# Gyroid Utils  
High-performance Python utilities for mesh processing, STL/STEP handling, OpenCascade operations, geometry cleanup, and 3D visualization.

This library combines the strengths of **NumPy**, **trimesh**, **Open3D**, **Plotly**, and **OpenCascade** into a unified, production-ready toolkit.  
Originally built for generating and analyzing **gyroid structures**, it is general-purpose and useful for any triangle-mesh workflow.

---

## 🚀 Features

### 🧩 Mesh Processing
- Extract largest connected component  
- Quadric mesh simplification (Open3D)  
- Fast triangle area computation (NumPy)  
- Mesh preview as interactive HTML (Plotly)  
- Histogram visualization of triangle sizes  

### 🏗 CAD / OpenCascade Tools
- Build planar faces from triangle data  
- Stitch faces into closed shells  
- Validate and simplify OCC shapes  
- Export shapes to STEP files  
- Check if a shell is watertight/closed  

### 📥 Input / Output
- Robust STL loading (handles ASCII + binary)  
- STEP export  
- Configurable logging for all operations  

---

## 📦 Installation

```bash
pip install gyroid_utils

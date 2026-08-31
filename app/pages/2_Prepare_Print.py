from dataclasses import dataclass
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st
import numpy as np

from gyroid_utils import voxel_tools
from gyroid_utils.mesh_tools import matrix_from_mesh, mesh_from_matrix

from app.components.tpms_source_panel import load_STL
from app.components.file_picker import browse_file
from app.components.mesh_preview import render_mesh_preview
from app.components.field_view import render_field_slice
from app.components.prepare_print_source import detect_overhangs, render_overhang_legend, minimize_overhangs

from app.components.tpms_source_panel import (load_STL,
    pad_to_square,)

# ============================================================
# ============== define internal variables ===================
# ============================================================
@dataclass
class page_parameters:
    structure_path: str = None
    verts: np.ndarray = None
    faces: np.ndarray = None
    geometry_matrix: np.ndarray = None
    resolution: int = 64
    X: np.ndarray = None
    Y: np.ndarray = None
    Z: np.ndarray = None
    overhang_matrix: np.ndarray = None
    R_geometry_matrix: np.ndarray = None
    R_X: np.ndarray = None
    R_Y: np.ndarray = None
    R_Z: np.ndarray = None
    R_verts: np.ndarray = None
    simplification_factor: float = 1.0
    max_faces: bool = False
    max_faces_count: int = 100000
    auto_smooth: bool = False
    smoothing_factor: float = 0.9

# Seed st.session_state only once, then always fetch the SAME persisted
# instance back out - re-running 'page_parameters = page_parameters()' every
# rerun (as before) made a brand-new, all-defaults object each time, and
# anything that later did st.session_state["page_parameters"] = page_parameters
# clobbered whatever had been computed on a previous run (e.g. R_geometry_matrix
# from clicking "Generate") the moment the user touched any other widget.
# Mutating fields on this object now mutates the persisted one directly (same
# object, not a copy), so no explicit write-back is needed anywhere below.
if "page_parameters" not in st.session_state:
    st.session_state["page_parameters"] = page_parameters()
page_parameters = st.session_state["page_parameters"]


# ============================================================
# ============== define internal functions ===================
# ============================================================
# this is done in one st.cache_data decorated function to avoid multiple reruns of the page when the user changes something in the UI.
@st.cache_data(show_spinner=False, max_entries=2)
def load_a_geometry_from_STL(stl_path: str, resolution: int):
    """
    Load a geometry from an STL file and convert it to a voxel matrix.
    """
    verts, faces = load_STL(stl_path)
    X, Y, Z, geometry_matrix = matrix_from_mesh(verts = verts, faces = faces, resolution = resolution)
    X, Y, Z = np.meshgrid(X, Y, Z, indexing='ij')
    return verts, faces, geometry_matrix, X, Y, Z

# ============================================================
# ===================== Start Page ===========================
# ============================================================
st.title("Prepare 3D print of TPMS")

col_params, col_preview = st.columns([1, 1.4])
# ============================================================
# ================== Select structure ========================
# ============================================================
with col_params:
    st.subheader("Select input file")
    # ----- select file -----
    browse_file(key = "structure_path",
        title="Select a file",
        filetypes=[("STL files", "*.stl"), ("All files", "*.*")],)
    page_parameters.structure_path = st.session_state["structure_path"]

    # ----- extract geometry -----
    if '.stl' in page_parameters.structure_path:  
        #st.warning("STL files must be converted to .npy format for processing. Please choose a resolution.")
        page_parameters.resolution = st.slider(
                "Grid resolution (per axis)", 16, 500, page_parameters.resolution, step=8,
                help="Higher = finer surface but slower generation. Start low (~48-64) while iterating.",
            )
        page_parameters.verts, page_parameters.faces, page_parameters.geometry_matrix, page_parameters.X, page_parameters.Y, page_parameters.Z = load_a_geometry_from_STL(page_parameters.structure_path, page_parameters.resolution)

    else:
        st.error("Please select a valid .stl file for the geometry.")
    st.divider()

with col_preview:
    st.subheader("Solid preview of input geometry")
    # ----- visualize geometry -----
    if page_parameters.verts is not None and page_parameters.faces is not None:
        render_mesh_preview(page_parameters.faces, page_parameters.verts, key="generate")
    
        st.checkbox("Show 2D slice view of voxelized geometry", key="show_2d_slice_view", value=False)
        if st.session_state["show_2d_slice_view"]:
            st.subheader("voxelized input geometry (2D slice)")
            if page_parameters.geometry_matrix is not None:
                render_field_slice(
                    page_parameters.geometry_matrix, page_parameters.X, page_parameters.Y, page_parameters.Z, key="generate",
                    value_range=(float(page_parameters.geometry_matrix.min()), float(page_parameters.geometry_matrix.max())),
                )
    else:
            st.info("Select a file first.")

# ============================================================
# ================= Show overhangs in solid ==================
# ============================================================
with col_params:
    if page_parameters.geometry_matrix is not None:
        col_1, col_2 = st.columns([1, 15], vertical_alignment="bottom")
        with col_1:
            show_overhangs = st.checkbox(" ", value=False,)
        with col_2:
            st.subheader("Show overhangs in input file")
    else:
        show_overhangs = False
        st.subheader("Show overhangs in input file")
        st.info("Select a file first.")


    if show_overhangs:
        angle_threshold = st.slider(
            "Overhang angle threshold (degrees)", 0, 90, 45, step=5, key="preview_overhang_angle_threshold",
            help="Overhangs are detected where the surface normal is more than this angle from vertical. Lower = more sensitive detection."
        )
        bridge_threshold = st.slider(
            "Bridge length threshold (mm)", 0, 100, 30, step=5, key="preview_overhang_bridge_threshold",
            help="Bridges are detected where the surface normal is more than this angle from vertical. Lower = more sensitive detection."
        )
        page_parameters.overhang_matrix = detect_overhangs(page_parameters.geometry_matrix, 
                                                           page_parameters.X, page_parameters.Y, page_parameters.Z, 
                                                           angle = angle_threshold, 
                                                           bridge=bridge_threshold, 
                                                           add_support_voxels=True)

        render_field_slice(
                        page_parameters.overhang_matrix, page_parameters.X, page_parameters.Y, page_parameters.Z, key="overhangs", value_range=(0, 4))
        render_overhang_legend()
    st.divider()

# ============================================================
# =============== calculate best orientation =================
# ============================================================
with col_params:
    if page_parameters.geometry_matrix is not None:
        st.subheader("Calculate best orientation for printing")
        angle_threshold = st.slider(
                    "Overhang angle threshold (degrees)", 0, 90, 45, step=5,
                    help="Overhangs are detected where the surface normal is more than this angle from vertical. Lower = more sensitive detection."
        )
        bridge_threshold = st.slider(
            "Bridge length threshold (mm)", 0, 100, 30, step=5,
            help="Bridges are detected where the surface normal is more than this angle from vertical. Lower = more sensitive detection."
        )
        complexity = st.slider(
            "Complexity", 1, 100, 25, step=1,
            help="The complexity of the optimization process. Higher = more accurate but slower."
        )
        generate = st.button(
            "Generate", type="primary", key = "generate_best_orientation", help="Calculate the best orientation for printing to minimize overhangs.")
    else:
        generate = False
        st.subheader("Calculate best orientation for printing")
        st.info("Select a file first.")
    if generate:
        # page_parameters IS st.session_state["page_parameters"] now (see top of
        # file) - setting fields on it here already persists them, no separate
        # write-back needed (and no 'else' branch to blank it back out).
        page_parameters.R_geometry_matrix, page_parameters.R_X, page_parameters.R_Y, page_parameters.R_Z, page_parameters.R_verts = minimize_overhangs(geometry_matrix = page_parameters.geometry_matrix, 
                       _X = page_parameters.X, _Y = page_parameters.Y, _Z = page_parameters.Z, 
                       complexity = complexity,
                       angle = angle_threshold,
                       bridge = bridge_threshold,
                       verts = page_parameters.verts)
        page_parameters.R_X, page_parameters.R_Y, page_parameters.R_Z = np.meshgrid(page_parameters.R_X, page_parameters.R_Y, page_parameters.R_Z, indexing='ij')

with col_preview:
    if page_parameters.R_geometry_matrix is not None:
        st.subheader("Best orientation for printing")
        render_field_slice(
            page_parameters.R_geometry_matrix, 
            page_parameters.R_X, page_parameters.R_Y, page_parameters.R_Z, 
            key="best_orientation_slices", 
            value_range=(float(page_parameters.R_geometry_matrix.min()), float(page_parameters.R_geometry_matrix.max())),
        )
        render_overhang_legend()
        st.subheader("Solid preview of best orientation")
        render_mesh_preview(page_parameters.faces, page_parameters.R_verts, key="generate_best_orientation_preview")

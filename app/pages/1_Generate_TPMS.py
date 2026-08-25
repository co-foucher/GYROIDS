"""
Generate a TPMS structure: pick a built-in surface type or paste a custom
equation, set periods/thickness/resolution/mesh options, preview the
result, and export an STL.

STATUS: functional.
"""
# Repo root isn't on sys.path by default - add it before importing
# anything under `app.*`. See app/_bootstrap.py for why this is inlined
# per-file rather than a shared import.
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dataclasses import dataclass

import numpy as np
import streamlit as st

from gyroid_utils.TPMS_classes import (
    GyroidModel, SchwartzPModel, DiamondModel, IWPModel, NeoviusModel,
    FischerKochSModel, FRDModel, LidinoidModel, SplitPModel,
)

from app.state import init_state, get_output_dir
from app.components.equation_input import render_equation_input
from app.components.mesh_preview import render_mesh_preview
from app.components.field_view import render_field_slice
from app.components.file_picker import browse_file, browse_directory
from app.components.import_TPMS_files import import_matrix_from_file

st.set_page_config(page_title="Generate TPMS", layout="wide")
init_state()

# ============================================================
# ============== define internal variables ===================
# ============================================================

# ---- Built-in TPMS types (label -> class) ------
BUILTIN_TYPES = {
    "Gyroid": GyroidModel,
    "Schwartz P": SchwartzPModel,
    "Diamond": DiamondModel,
    "I-WP": IWPModel,
    "Neovius": NeoviusModel,
    "Fischer-Koch S": FischerKochSModel,
    "F-RD": FRDModel,
    "Lidinoid": LidinoidModel,
    "Split-P": SplitPModel,
}


# ---- Field modes (label -> TPMSModel.compute_field(mode=...) argument) ----
from app.components.tpms_source_panel import (
    render_field_mode,
    render_threshold,
    render_thickness,
    load_STL,
    generate_ui_tpms,
)


@dataclass
class TPMSParams:
    """
    Bundles the generation settings shared across all three "Surface"
    sources (Built-in type / Custom equation / Import from file), with
    their default values in one place. Per-source-only inputs - the
    periods (px/py/pz), the built-in type_name, and the raw equation
    string - stay as local variables in their own branch below, since
    only one branch at a time ever uses them.
    """
    size_x: float = 10.0
    size_y: float = 10.0
    size_z: float = 10.0
    resolution: int = 64
    thickness: float = 1.0
    field_mode: str = "distance"
    threshold: float = 0.0
    baseplate_thickness: float = 0.0
    simplification_factor: float = 0.9
    max_faces: bool = False
    max_faces_count: int = 100_000
    auto_smooth: bool = True
    smoothing_factor: float = 0.9


# ============================================================
# ============== define internal functions ===================
# ============================================================

def _pad_to_square(matrix, pad_value=0):
    target = max(matrix.shape)
    pad_width = [(0, target - dim) for dim in matrix.shape]
    return np.pad(matrix, pad_width=pad_width, mode="constant", constant_values=pad_value)

# ============================================================
# ===================== Start Page ===========================
# ============================================================
st.title("Generate a TPMS structure")

col_params, col_preview = st.columns([1, 1.4])

# ==========================================================
# ============== user defined parameters ===================
# ==========================================================
params = TPMSParams()  # fresh instance each rerun; each widget below overwrites
                        # its own field with its own persisted value (Streamlit
                        # keeps that per-widget, independent of this object)
with col_params:
    # ------ grid parameters ------
    st.subheader("Grid parameters")
    params.resolution = st.slider(
        "Grid resolution (per axis)", 16, 500, params.resolution, step=8,
        help="Higher = finer surface but slower generation. Start low (~48-64) while iterating.",
    )
    d1, d2, d3 = st.columns(3)
    params.size_x = d1.number_input("Size X", value=params.size_x, min_value=0.01)
    params.size_y = d2.number_input("Size Y", value=params.size_y, min_value=0.01)
    params.size_z = d3.number_input("Size Z", value=params.size_z, min_value=0.01)
    st.divider()

    # ------ choose TPMS type / paste equation / import from file ------
    st.subheader("TPMS Definition")
    source = st.radio("Surface", ["Built-in type", "Custom equation", "Import from file"], horizontal=True)
    equation = None
    type_name = None
    # Pre-initialized so generate_ui_tpms() below always receives bound
    # names, even for the branches the current "Surface" selection doesn't use.
    px = py = pz = None
    thickness = None
    field = None
    thickness_value = None

    # ---- Built-in type ----
    if source == "Built-in type":
        type_name = st.selectbox("TPMS type", list(BUILTIN_TYPES.keys()))
        c1, c2, c3 = st.columns(3)
        px = c1.number_input("Period X", value=5.0, min_value=0.01)
        py = c2.number_input("Period Y", value=5.0, min_value=0.01)
        pz = c3.number_input("Period Z", value=5.0, min_value=0.01)

        st.divider()
        params.field_mode = render_field_mode()
        params.threshold = render_threshold(params.field_mode)
        params.thickness = render_thickness(params.field_mode)

    # ---- Custom equation ----
    elif source == "Custom equation":
        equation, thickness = render_equation_input()
        params.field_mode = render_field_mode()
        params.threshold = render_threshold(params.field_mode)
        render_thickness(params.field_mode, draw_widget=False,
            thickness_source_desc="the Thickness formula above")

    # ---- Import from file ----
    # (after field/thickness_value are loaded from disk)
    elif source == "Import from file":
        st.session_state.setdefault("field_matrix_path", "")
        st.session_state.setdefault("thickness_matrix_path", "")
        # define TPMS field
        col_path, col_browse = st.columns([5, 1.5], vertical_alignment="bottom")
        with col_path:
            matrix_path = st.text_input("Path to matrix file", key="field_matrix_path")
        with col_browse:
            st.write("")  # spacer so the button lines up with the text box, not its label
            browse_file("field_matrix_path",
                title="Select a matrix file",
                filetypes=[("Numpy files", "*.npy"), ("CSV files", "*.csv"), ("All files", "*.*")],)
        field = import_matrix_from_file(file_path = st.session_state["field_matrix_path"])

        #define thickness field
        col_path, col_browse = st.columns([5, 1.5], vertical_alignment="bottom")
        with col_path:
            matrix_path = st.text_input("Path to thickness file", key="thickness_matrix_path")
        with col_browse:
            st.write("")  # spacer so the button lines up with the text box, not its label
            browse_file("thickness_matrix_path",
                title="Select a matrix file",
                filetypes=[("Numpy files", "*.npy"), ("CSV files", "*.csv"), ("All files", "*.*")],)
        thickness_value = import_matrix_from_file(file_path = st.session_state["thickness_matrix_path"])
        params.field_mode = render_field_mode()
        params.threshold = render_threshold(params.field_mode)
        render_thickness(params.field_mode, draw_widget=False,
            thickness_source_desc="the imported thickness file above")
    st.divider()

    # ----- add baseplate ------
    st.subheader("Baseplates")
    params.baseplate_thickness = st.number_input("Baseplate thickness (0 = none)", value=params.baseplate_thickness, min_value=0.0)
    st.divider()

    # ----- Combine with geometry ------
    col_1, col_2 = st.columns([1, 15], vertical_alignment="bottom")
    with col_1:
        combine_with_geometry = st.checkbox(" ", value=False,)
    with col_2:
        st.subheader("Combine with existing geometry")
    if combine_with_geometry:
        st.info("This feature is not yet implemented. In the future, you will be able to import a mesh or voxel grid and combine it with the generated TPMS structure.")
        col_path, col_browse = st.columns([5, 1.5], vertical_alignment="bottom")
        with col_path:
            matrix_path = st.text_input("Path to geometry file", key="combined_geometry_path")
        with col_browse:
            st.write("")  # spacer so the button lines up with the text box, not its label
            browse_file("combined_geometry_path",
                title="Select a matrix file",
                filetypes=[("STL files", "*.stl"), ("Numpy files", "*.npy"), ("All files", "*.*")],)
        if '.npy' in st.session_state["combined_geometry_path"]:
            geometry = import_matrix_from_file(file_path = st.session_state["combined_geometry_path"])
            geometry = _pad_to_square(geometry)
        elif '.stl' in st.session_state["combined_geometry_path"]:
            from gyroid_utils.mesh_tools import matrix_from_mesh
            st.session_state["combined_geometry_path"]
            verts, faces = load_STL(st.session_state["combined_geometry_path"])
            _,_,_, geometry = matrix_from_mesh(verts, faces, params.resolution)
            geometry = _pad_to_square(geometry)
        else:
            st.error("Please select a valid .npy or .stl file for the geometry.")
        combination_type = st.selectbox("Combination type", ["Intersection", "Union", "Substraction"])
        with col_preview:
            st.subheader("combined geometry preview")
            render_mesh_preview(faces, verts, key="c_geometry")

    else:
        geometry = None
        combination_type = None


    st.divider()

    # ----- mesh parameters ------
    st.subheader("Mesh parameters")
    params.simplification_factor = st.slider(
        "Mesh simplification (fraction of faces kept)", 0.1, 1.0, params.simplification_factor,
        help="Passed to TPMSModel.simplify_mesh(target_faces=...).",)
    params.max_faces = st.checkbox("Limit maximum faces", value=params.max_faces,
        help="If checked, the mesh is simplified to a maximum number of faces.")
    if params.max_faces:
        params.max_faces_count = st.number_input(
            "Maximum faces", value=params.max_faces_count, min_value=1,
            help="If 'Limit maximum faces' is checked, the mesh is simplified to this many faces.",)
    params.auto_smooth = st.checkbox("Auto-smooth mesh", value=params.auto_smooth,
        help="If checked, the mesh is smoothed after simplification and again after fixing.")
    if params.auto_smooth:
        params.smoothing_factor = st.slider(
            "Smoothing factor", 0.0, 1.0, params.smoothing_factor, step=0.01,
            help="Passed to TPMSModel.smooth_mesh(smoothing_factor=...). Higher = more smoothing.")
    # ----- generate button ------
    generate = st.button(
        "Generate", type="primary",
        disabled=(source == "Custom equation" and equation is None),)


# ==========================================================
# ===================== generate TPMS ======================
# ==========================================================
if generate:
    generate_ui_tpms(
        source=source, 
        params=params, 
        BUILTIN_TYPES=BUILTIN_TYPES,
        type_name=type_name, 
        px=px, py=py, pz=pz,
        custom_equation=equation, 
        custom_thickness=thickness,
        field=field, 
        thickness_value=thickness_value,
        geometry=geometry,
        combination_type = combination_type,
    )

model = st.session_state.get("current_model")


# ==========================================================
# =================== preview section ======================
# ==========================================================
with col_preview:
    st.subheader("IMPLICIT Field (2D slice)")
    if model is not None and model.implicit_field is not None:
        render_field_slice(
            model.implicit_field, model.x, model.y, model.z, key="generate",
            value_range=st.session_state.get("current_field_range"),
        )
    else:
        st.info("Compute a field first to see the 2D slice view.")

    st.subheader("Mesh preview")
    if model is not None and model.faces is not None:
        render_mesh_preview(model.faces, model.verts, key="generate")
    else:
        st.info("Set parameters and click Generate.")



# ==========================================================
# ================ save outputs section ====================
# ==========================================================
with col_preview:
    if model is not None and model.faces is not None:
        name = st.text_input("File name", value="my_tpms")

        # ----- Export STL ------
        if st.button("Export STL"):
            out_path = get_output_dir() / name
            model.export_stl(str(out_path))
            model.save_mesh_preview(str(out_path))  # keep an .html alongside the .stl, picked up by the Library page
            st.success(f"Saved {out_path}.stl (+ preview .html)")

        # ----- Export TPMS field ------
        if st.button("Export TPMS implicit field (.npy)"):
            out_path = get_output_dir() / name
            if model.implicit_field is None: # check if  field exists
                st.warning("No TPMS field available to export. Please ensure that the model field exists before exporting.")
            np.save(str(out_path) + ".npy", model.implicit_field)
            st.success(f"Saved {out_path}.npy")

        # ----- Export TPMS thickness field ------
        if st.button("Export TPMS thickness field (.npy)"):
            out_path = get_output_dir() / (name + "_thickness")
            if model.thickness is None: # check if thickness field exists
                st.error("No thickness field available to export. Please ensure that the model has a thickness field before exporting.")
            elif isinstance(model.thickness, float): #check if thickness field is a float
                st.warning("Thickness field is a float, not an array...")
                np.save(str(out_path) + ".npy", np.array([model.thickness])) #save as a 1D array
                st.success(f"Saved {out_path}.npy")
            elif isinstance(model.thickness, np.ndarray) and model.thickness.ndim == 3: #check if thickness field is a 3D array
                np.save(str(out_path) + ".npy", model.thickness)
                st.success(f"Saved {out_path}.npy")
            else:
                st.error("Thickness field is not a valid 3D array. Please ensure that the model's thickness field is a valid 3D array before exporting.")
    else:
        st.info("Set parameters and click Generate.")

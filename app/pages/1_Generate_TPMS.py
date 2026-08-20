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

import numpy as np
import streamlit as st

from gyroid_utils.TPMS_classes import (
    GyroidModel, SchwartzPModel, DiamondModel, IWPModel, NeoviusModel,
    FischerKochSModel, FRDModel, LidinoidModel, SplitPModel,
)
from gyroid_utils.TPMS_classes.tpms_custom import CustomTPMSModel

from app.state import init_state, get_output_dir
from app.components.equation_input import render_equation_input, evaluate_custom_inputs, EquationError
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
# mode use to calculate the density field from the implicit field.
FIELD_MODES = {
    "Distance": "distance",
    "Signed": "signed",
    "Signed (inverted)": "signed_inverse",
    "Band": "band",
}

FIELD_HELPS = {
    "Distance": "The density field is computed as the distance (defined by the thickness) to an iso-surface (defined by the threshold).",
    "Signed": "The density field is computed by implicit_field > threshold.",
    "Signed (inverted)": "The density field is computed by implicit_field < threshold.",
    "Band": "The density field is computed as density_field = (thickness - np.abs(implicit_field)) > threshold. This is a fast approximation of the distance field",
}

# ----- level / thickness modes ----
# field modes are separated into two categories: those that use a `level` combined with a thickness (find the surface at the given level, and give it a thickness) 
# and those that use only a `threshold` (the implicit field value at which the mesh is extracted). 
LEVEL_MODES = ("signed", "signed_inverse")
THICKNESS_MODES = ("band", "distance", "distance_fast")

# ============================================================
# ============== define internal functions ===================
# ============================================================

# it would be nice to try and do a few functions to make the code below more readable, 

# ============================================================
# ===================== Start Page ===========================
# ============================================================
st.title("Generate a TPMS structure")

col_params, col_preview = st.columns([1, 1.4])

# ==========================================================
# ============== user defined parameters ===================
# ==========================================================
with col_params:
    # ------ grid parameters ------
    st.subheader("Grid parameters")
    resolution = st.slider(
        "Grid resolution (per axis)", 16, 500, 64, step=8,
        help="Higher = finer surface but slower generation. Start low (~48-64) while iterating.",
    )
    d1, d2, d3 = st.columns(3)
    size_x = d1.number_input("Size X", value=10.0, min_value=0.01)
    size_y = d2.number_input("Size Y", value=10.0, min_value=0.01)
    size_z = d3.number_input("Size Z", value=10.0, min_value=0.01)
    st.divider()

    # ------ choose TPMS type / paste equation / import from file ------
    st.subheader("TPMS Definition")
    source = st.radio("Surface", ["Built-in type", "Custom equation", "Import from file"], horizontal=True)
    equation = None
    type_name = None

    if source == "Built-in type":
        type_name = st.selectbox("TPMS type", list(BUILTIN_TYPES.keys()))
        c1, c2, c3 = st.columns(3)
        px = c1.number_input("Period X", value=5.0, min_value=0.01)
        py = c2.number_input("Period Y", value=5.0, min_value=0.01)
        pz = c3.number_input("Period Z", value=5.0, min_value=0.01)

        mode_label = st.selectbox("Field mode", list(FIELD_MODES.keys()), index=0, 
                                  key="field_mode",
                                  help = FIELD_HELPS[st.session_state.get("field_mode", "Distance")])
        field_mode = FIELD_MODES[mode_label]

        if field_mode in LEVEL_MODES:
            threshold = st.number_input("Field threshold (implicit field minimum value defining the solid)", value=0.0,
                help="The TPMS surface is placed where the implicit field equals this value. Every voxel with implicit_field > level is considered solid, and the thickness is applied to that solid region.")
        else:
            threshold = st.number_input("Field threshold (for surface extraction)", value=0.0,
                help="The field isosurface at this value is extracted to generate a surface, that is then thickened.")

        if field_mode in THICKNESS_MODES:
            thickness = st.number_input("Thickness", value=1.0, min_value=0.05)
        else:
            thickness = 0.0

    elif source == "Custom equation":
        equation, thickness = render_equation_input()

        mode_label = st.selectbox("Field mode", list(FIELD_MODES.keys()), index=0)
        field_mode = FIELD_MODES[mode_label]
        if field_mode not in THICKNESS_MODES:
            st.caption(
                "Note: the Thickness formula above is ignored in "
                f"'{mode_label}' mode - it doesn't use thickness at all."
            )

        if field_mode in LEVEL_MODES:
            threshold = st.number_input("Field threshold (implicit field minimum value defining the solid)", value=0.0,
                help="The TPMS surface is placed where the implicit field equals this value. Every voxel with implicit_field > level is considered solid, and the thickness is applied to that solid region.")
        else:
            threshold = st.number_input("Field threshold (for surface extraction)", value=0.0,
                help="The field isosurface at this value is extracted to generate a surface, that is then thickened.")

    elif source == "Import from file":
        st.warning("File import functionality is not yet implemented.")
        st.session_state.setdefault("field_matrix_path", "")
        st.session_state.setdefault("thickness_matrix_path", "")
        # define TPMS field
        col_path, col_browse = st.columns([5, 1])
        with col_path:
            matrix_path = st.text_input("Path to matrix file", key="field_matrix_path")
        with col_browse:
            st.write("")  # spacer so the button lines up with the text box, not its label
            browse_file(
                "field_matrix_path",
                title="Select a matrix file",
                filetypes=[("Numpy files", "*.npy"), ("CSV files", "*.csv"), ("All files", "*.*")],)
        field = import_matrix_from_file(file_path = st.session_state["field_matrix_path"])

        #define thickness field
        col_path, col_browse = st.columns([5, 1])
        with col_path:
            matrix_path = st.text_input("Path to thickness file", key="thickness_matrix_path")
        with col_browse:
            st.write("")  # spacer so the button lines up with the text box, not its label
            browse_file(
                "thickness_matrix_path",
                title="Select a matrix file",
                filetypes=[("Numpy files", "*.npy"), ("CSV files", "*.csv"), ("All files", "*.*")],)
        thickness_value = import_matrix_from_file(file_path = st.session_state["thickness_matrix_path"])

        mode_label = st.selectbox("Field mode", list(FIELD_MODES.keys()), index=0)
        field_mode = FIELD_MODES[mode_label]
        if field_mode not in THICKNESS_MODES:
            st.caption(
                "Note: the imported thickness file above is ignored in "
                f"'{mode_label}' mode - it doesn't use thickness at all."
            )

        if field_mode in LEVEL_MODES:
            threshold = st.number_input("Field threshold (implicit field minimum value defining the solid)", value=0.0,
                help="The TPMS surface is placed where the implicit field equals this value. Every voxel with implicit_field > level is considered solid, and the thickness is applied to that solid region.")
        else:
            threshold = st.number_input("Field threshold (for surface extraction)", value=0.0,
                        help="The field isosurface at this value is extracted to generate a surface, that is then thickened.")
    st.divider()

    # ----- additional features ------
    st.subheader("Additional Features")
    baseplate_thickness = st.number_input("Baseplate thickness (0 = none)", value=0.0, min_value=0.0)
    st.divider()
    
    # ----- mesh parameters ------
    st.subheader("Mesh parameters")
    simplification_factor = st.slider(
        "Mesh simplification (fraction of faces kept)", 0.1, 1.0, 0.9,
        help="Passed to TPMSModel.simplify_mesh(target_faces=...).",
    )
    max_faces = st.checkbox("Limit maximum faces", value=False,
        help="If checked, the mesh is simplified to a maximum number of faces.")
    if max_faces:
        simplification_factor = st.number_input(
            "Maximum faces", value=100_000, min_value=1,
            help="If 'Limit maximum faces' is checked, the mesh is simplified to this many faces.",
        )
    auto_smooth = st.checkbox("Auto-smooth mesh", value=True,
        help="If checked, the mesh is smoothed after simplification and again after fixing.")
    if auto_smooth:
        smoothing_factor = st.slider(
            "Smoothing factor", 0.0, 1.0, 0.9, step=0.01,
            help="Passed to TPMSModel.smooth_mesh(smoothing_factor=...). Higher = more smoothing.")
    # ----- generate button ------
    generate = st.button(
        "Generate", type="primary",
        disabled=(source == "Custom equation" and equation is None),
    )


# ==========================================================
# ===================== generate TPMS ======================
# ==========================================================
if generate:
    x, y, z = np.meshgrid(
        np.linspace(0, size_x, resolution),
        np.linspace(0, size_y, resolution),
        np.linspace(0, size_z, resolution),
        indexing="ij",
    )

    try:
        with st.spinner("Computing field and generating mesh..."):

            # ----- Built-in type ------ 
            if source == "Built-in type":
                model = BUILTIN_TYPES[type_name](x, y, z, px, py, pz, thickness)
            
            # ----- Custom equation ------ 
            elif source == "Custom equation":
                # equation_input.evaluate_custom_inputs() turns the two
                # strings into plain arrays on the real generation grid -
                # this page has no parsing-related imports at all, and
                # CustomTPMSModel has no px/py/pz to pass (see its
                # docstring).
                field, thickness_value = evaluate_custom_inputs(equation, thickness, x, y, z)
                model = CustomTPMSModel(x, y, z, thickness_value, field=field)
            
            # ----- Import from file ------ 
            elif source == "Import from file":
                if field.shape != (resolution, resolution, resolution):
                    st.warning(f"Imported field shape {field.shape} does not match the expected resolution ({resolution}, {resolution}, {resolution}).")
                if thickness_value.ndim == 1:
                    thickness_value = thickness_value[0]  # take the first value as a scalar thickness
                elif thickness_value.shape != (resolution, resolution, resolution) and not np.isscalar(thickness_value):
                    st.warning(f"Imported thickness shape {thickness_value.shape} does not match the expected resolution ({resolution}, {resolution}, {resolution}).")
                model = CustomTPMSModel(x, y, z, thickness_value, field=field)
            
            # "signed"/"signed_inverse"/"distance"/"distance_fast" bake the
            # GUI's threshold value into compute_field() as the reference
            # `level` (the surface sits at implicit_field == level), so the
            # mesh extraction below always happens at the fixed iso_level
            # 0.0 for those modes. "band" has no `level` concept - its
            # threshold is passed straight through as the mesh iso_level,
            # same as before this mode overhaul.
            if field_mode in LEVEL_MODES:
                model.compute_field(mode=field_mode, level=threshold)
                mesh_iso_level = 0.0
            else:
                model.compute_field(mode=field_mode)
                mesh_iso_level = threshold

            # ----- add baseplates ------
            if baseplate_thickness > 0:
                model.add_baseplates(thickness=baseplate_thickness)

            # ----- generate mesh ------
            st.session_state["current_field_range"] = (float(model.implicit_field.min()), float(model.implicit_field.max()))
            model.generate_mesh(iso_level=mesh_iso_level)
            model.simplify_mesh(target_faces=simplification_factor)
            if auto_smooth:
                model.smooth_mesh(smoothing_factor=smoothing_factor)
            model.fix_mesh()
            is_valid = model.check_mesh_quality()

        st.session_state["current_model"] = model
        st.session_state["current_equation"] = equation

        if not is_valid:
            st.warning(
                "Generated mesh failed validity checks (not watertight / "
                "self-intersecting). Try a coarser grid, a different "
                "thickness, or a different field mode."
            )
        else:
            st.success(f"Mesh generated: {len(model.faces)} faces.")
    except EquationError as e:
        st.error(f"Equation error: {e}")

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
            if model.implicit_field is None: # check if thickness field exists
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

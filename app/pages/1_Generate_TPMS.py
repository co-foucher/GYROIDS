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

st.set_page_config(page_title="Generate TPMS", layout="wide")
init_state()

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

st.title("Generate a TPMS structure")

col_params, col_preview = st.columns([1, 1.4])

# ==========================================================
# ============== user defined parameters ===================
# ==========================================================
with col_params:
    # ------ choose TPMS type or paste equation ------
    source = st.radio("Surface", ["Built-in type", "Custom equation"], horizontal=True)
    equation = None
    type_name = None
    if source == "Built-in type":
        type_name = st.selectbox("TPMS type", list(BUILTIN_TYPES.keys()))
    else:
        equation, thickness = render_equation_input()
        field_mode = st.selectbox("Field mode", ["distance", "signed", "abs"], index=0)
        threshold = st.number_input("Field threshold (for surface extraction)", value=0.0,
            help="The field isosurface at this value is extracted to generate the mesh.")


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

    # ------ TPMS parameters ------
    if source == "Built-in type":
        st.subheader("TPMS parameters")
        c1, c2, c3 = st.columns(3)
        px = c1.number_input("Period X", value=5.0, min_value=0.01)
        py = c2.number_input("Period Y", value=5.0, min_value=0.01)
        pz = c3.number_input("Period Z", value=5.0, min_value=0.01)
        thickness = st.number_input("Thickness", value=1.0, min_value=0.05)
        field_mode = st.selectbox("Field mode", ["distance", "signed", "abs"], index=0)
        threshold = st.number_input("Field threshold (for surface extraction)", value=0.0,
            help="The field isosurface at this value is extracted to generate the mesh.")
    # ----- additional features ------
    st.subheader("Additional Features")
    baseplate_thickness = st.number_input("Baseplate thickness (0 = none)", value=0.0, min_value=0.0)
    
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
            if source == "Built-in type":
                model = BUILTIN_TYPES[type_name](x, y, z, px, py, pz, thickness)
            else:
                # equation_input.evaluate_custom_inputs() turns the two
                # strings into plain arrays on the real generation grid -
                # this page has no parsing-related imports at all, and
                # CustomTPMSModel has no px/py/pz to pass (see its
                # docstring).
                field, thickness_value = evaluate_custom_inputs(equation, thickness, x, y, z)
                model = CustomTPMSModel(x, y, z, thickness_value, field=field)

            model.compute_field(mode=field_mode)
            if baseplate_thickness > 0:
                model.add_baseplates(thickness=baseplate_thickness)
            # cache the field's value range once here (after baseplates, which
            # edit self.v in place) rather than recomputing it on every 2D-slice
            # slider drag below - v can be large at high resolution (up to
            # 500^3 with the current slider max).
            st.session_state["current_field_range"] = (float(model.v.min()), float(model.v.max()))
            model.generate_mesh(iso_level=threshold)
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
    st.subheader("Field (2D slice)")
    if model is not None and model.v is not None:
        render_field_slice(
            model.v, model.x, model.y, model.z, key="generate",
            value_range=st.session_state.get("current_field_range"),
        )
    else:
        st.info("Compute a field first to see the 2D slice view.")

    st.subheader("Mesh preview")
    if model is not None and model.faces is not None:
        render_mesh_preview(model.faces, model.verts, get_output_dir(), key="generate")
        name = st.text_input("File name", value="my_tpms")
        if st.button("Export STL"):
            out_path = get_output_dir() / name
            model.export_stl(str(out_path))
            model.save_mesh_preview(str(out_path))  # keep an .html alongside the .stl, picked up by the Library page
            st.success(f"Saved {out_path}.stl (+ preview .html)")
    else:
        st.info("Set parameters and click Generate.")

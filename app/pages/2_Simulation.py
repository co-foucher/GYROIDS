"""
Mesh an STL with fTetWild, then create/run an ABAQUS simulation from it.

STATUS: scaffold. Wires the real gyroid_utils.TET_mesh_tools /
abaqus_tools calls into the UI with a minimal set of parameters (matching
examples/generate_frequency_sim.py and notebooks/full simulation
workflow.ipynb) - extend the forms below as more simulation types/options
are needed. Both external calls are long-running, so they run in a
background thread via app.jobs (see that module's docstring for why).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st
import shutil
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(page_title="Simulation", layout="wide")

# Heavy imports (meshio/mesh/vtk/plotly stack via gyroid_utils) live behind
# a spinner so the page shows something immediately instead of appearing
# frozen on first load. Cached after the first import - see
# src/gyroid_utils/__init__.py.
with st.spinner("Loading GYROIDS toolkit..."):
    from gyroid_utils import TET_mesh_tools, abaqus_tools

    from app.state import init_state, get_output_dir
    from app.components.jobs import start_job, render_job_status
    from app.components.Simulation_source import mesh_job, create_abaqus_job, run_abaqus_job, render_load_case_form
    from app.components.file_picker import browse_file, browse_directory
    from app.components.tpms_source_panel import load_STL
    from app.components.mesh_preview import render_mesh_preview
    from app.components.load_preview import render_load_preview

init_state()
st.title("Mesh + Simulation")

# ===============================================================
# ================== Tetrahedral meshing ========================
# ===============================================================
default_dir = str(get_output_dir())
st.subheader("1. Tetrahedral meshing (fTetWild)")
col_1, col_2 = st.columns([1, 1.4])
with col_1:
    # ------  select the input STL ------
    browse_file(key = "structure_path",
        title="Select an STL file for simulation",
        filetypes=[("STL files", "*.stl"), ("All files", "*.*")],)
    stl_dir = st.session_state["structure_path"]
    file_name = stl_dir.split("/")[-1].split(".")[0] if stl_dir else None
    stl_dir = "/".join(stl_dir.split("/")[:-1]) if stl_dir else None

    # ------ inform fTetWild parameters ------
    ftetwild_path = st.text_input(
        "fTetWild executable path",
        value=r"C:\Program Files\fTetWild\build\Release\FloatTetwild_bin.exe",
    )
    c1, c2 = st.columns(2)
    epsilon = c1.number_input("Epsilon (envelope size)", value=0.001, format="%.5f")
    cpu_cores = c2.number_input("CPU cores", value=1, min_value=1, step=1)

    # ------ run fTetWild meshing ------
    if st.button("Run fTetWild meshing"):
        # if the user hasn't selected an STL file, show an error
        if not stl_dir or not file_name:
            st.error("Please select an STL file first.")
        # if another meshing job is already running, don't start a new one
        if st.session_state["jobs"].get(st.session_state.get("mesh_job_id")) is not None:
            if st.session_state["jobs"].get(st.session_state.get("mesh_job_id")).status == "running":
                st.warning("Meshing job is already running. Please wait for it to finish.")
        else :
            job_id = start_job(st.session_state["jobs"],
                            f"mesh:{file_name}",
                            mesh_job,
                            stl_dir=stl_dir,
                            file_name=file_name,
                            ftetwild_path=ftetwild_path,
                            epsilon=epsilon,
                            cpu_cores=cpu_cores)
            st.session_state["mesh_job_id"] = job_id

    render_job_status(st.session_state["jobs"], "mesh_job_id")

# kept at page scope so section 2's load-case preview can reuse them instead
# of re-reading the STL from disk on every rerun.
verts, faces = None, None
with col_2:
    if stl_dir is not None:
        verts, faces = load_STL(st.session_state["structure_path"])
        render_mesh_preview(faces, verts, key="generate")
    else:
        st.info("Select a file to preview.")
st.divider()


# ======================================================================
# ==================== create ABAQUS simulation ========================
# ======================================================================
# those are the built-in simulation scripts that can be selected from the dropdown menu.
# they are stored in the gyroid_utils/pybaqus folder.

BUILTIN_SIM = {
    "Frequency analysis": "generate_frequency_sim.py",
    "Static analysis": "generate_static_sim.py",
}
BUILTIN_SIM_INFO = {
    "Frequency analysis": "Extract the first 10 natural frequencies of the structure.",
    "Static analysis": (
        "Linear elastic uniaxial load case, for measuring the structure's "
        "stiffness in one direction.\n\n"
        "The two extreme faces along the chosen axis are found automatically "
        "(the slab of nodes within the tolerance below of the bounding-box "
        "min/max). The bottom face is held **only** along the load axis - a "
        "roller, so the structure stays free to expand sideways and you get the "
        "lattice's own stiffness rather than a platen-confined one. Two single "
        "nodes on that face are pinned in-plane to remove the leftover "
        "rigid-body motion. The total load you enter is then split equally over "
        "every node of the top face."
    ),
}
# simulation types that take a load case on top of the material properties
LOAD_CASE_SIM = {"Static analysis"}

st.subheader("2. create ABAQUS simulation")
@st.fragment
def create_simulation_section():
    col_1, col_2 = st.columns([1, 1.4])
    with col_1:
        # -------  select the simulation type ------
        script_accro = st.selectbox("Simulation type", list(BUILTIN_SIM.keys()))
        script_name = BUILTIN_SIM[script_accro]
        # ------  select if quadratic mesh is needed ------
        quadratic_mesh = st.toggle(
                "Use quadratic elements",
                help="Use higher-order elements for more accurate results.",
            )
        quadratic_mesh = 1 if quadratic_mesh else 0

        # ------ give material properties for the simulation ------
        st.write("Material properties for the simulation :")
        st.caption("ABAQUS style - you have to make sure the units are consistent with the STL file's units.")
        young_modulus = st.number_input("Young's Modulus", value=300000.0, format="%.1f")
        poisson_ratio = st.number_input("Poisson's Ratio", value=0.21, format="%.2f")
        density = st.number_input("Density", value=3.9e-09, format="%.2e")

        # ------ load case (static analysis only) ------
        load_case = None
        if script_accro in LOAD_CASE_SIM:
            st.write("Load case :")
            load_case = render_load_case_form(key="static") #here this function output a dictionnary with the load case parameters, which will be passed to the abaqus script.
    # because these variables are created inside a function, you have put them in the session state so that the rest of the code can access them.
    st.session_state["sim_config"] = {
        "script_name": script_name,
        "young_modulus": young_modulus,
        "poisson_ratio": poisson_ratio,
        "density": density,
        "quadratic_mesh": quadratic_mesh,
        "load_case": load_case,
    }

    with col_2:
        st.write("Simulation type info :")
        st.caption(BUILTIN_SIM_INFO[script_accro])
        if script_accro in LOAD_CASE_SIM:
            # second preview of the same STL, this time showing where the load and
            # the supports land - cheap sanity check before a long solve.
            rbm_coord = render_load_preview(mesh_name=file_name,
                                faces=faces,
                                verts=verts,
                                case=load_case,
                                key="static")
            if rbm_coord is not None:
                temp = [str(x) for x in rbm_coord[0,:]]
                st.session_state["sim_config"]["load_case"]["pin_a_coords"] = temp
                st.warning(f"Pin A coordinates: {temp}")
                temp = [str(x) for x in rbm_coord[1,:]]
                st.session_state["sim_config"]["load_case"]["pin_b_coords"] = temp
                st.warning(f"Pin B coordinates: {temp}")
    




create_simulation_section()

@st.fragment
def render_create_abaqus_section(stl_dir: str, file_name: str) -> None:
    if st.button("Create ABAQUS simulation input"):
        if not stl_dir or not file_name:
            st.error("Please select an STL file first.")
        elif "sim_config" not in st.session_state:
            st.error("Set the simulation type / material properties above first.")
        else:
            # grab the simulation config from the session state, which was set by the create_simulation_section() function above.
            cfg = st.session_state["sim_config"]
            script_name = cfg["script_name"]

            # create a sub folder for the simulation output files, named after the STL file
            sim_output_dir = Path(stl_dir) / f"{file_name}_sim"
            sim_output_dir.mkdir(parents=True, exist_ok=True)
            # fetch the simulation script from the built-in scripts and write it to the output folder
            pybaqus_dir = Path(abaqus_tools.__file__).parent / "pybaqus"
            script_path = str(sim_output_dir / script_name)
            shutil.copy(pybaqus_dir / script_name, script_path)

            # create a job to run the abaqus simulation in a background thread, passing the output folder and script name as arguments
            job_id = start_job(st.session_state["jobs"],
                                f"abaqus:{file_name}",
                                create_abaqus_job,
                                stl_dir=stl_dir,
                                file_name=file_name,
                                script_dir=sim_output_dir,
                                script_name=script_path,
                                young_modulus=cfg["young_modulus"],
                                poisson_ratio=cfg["poisson_ratio"],
                                density=cfg["density"],
                                quadratic_tets=cfg["quadratic_mesh"],
                                extra_args=cfg["load_case"])
            st.session_state["abaqus_job_id"] = job_id              # you save it to be able to use it in the results section.
            st.session_state["sim_output_dir"] = sim_output_dir     # you save it to be able to use it in the results section.


render_create_abaqus_section(stl_dir, file_name)
render_job_status(st.session_state["jobs"], "abaqus_job_id")

st.divider()

# ======================================================================
# ====================== run ABAQUS simulation =========================
# ======================================================================

@st.fragment
def render_run_abaqus_section(stl_dir: str, file_name: str) -> None:
    st.subheader("3. Run ABAQUS simulation")

    CPU_CORES = st.slider("CPU cores for ABAQUS", min_value=1, max_value=16, value=1, step=1)

    if st.button("Run ABAQUS simulation"):
        # if the user hasn't selected an STL file, show an error
        if not stl_dir or not file_name:
            st.error("Please select an STL file first.")
        else:
            # same folder create_abaqus_job wrote the simulation input into
            sim_output_dir = Path(stl_dir) / f"{file_name}_sim"
            # if a run is already in progress for this session, don't start another one
            run_job = st.session_state["jobs"].get(st.session_state.get("run_job_id"))
            if run_job is not None and run_job.status == "running":
                st.warning("A simulation run is already in progress. Please wait for it to finish.")
            else:
                job_id = start_job(st.session_state["jobs"],
                                    f"run:{file_name}",
                                    run_abaqus_job,
                                    file_name=file_name,
                                    sim_dir=sim_output_dir)
                st.session_state["run_job_id"] = job_id

render_run_abaqus_section(stl_dir, file_name)
render_job_status(st.session_state["jobs"], "run_job_id")



# ======================================================================
# ====================== extract results from ODB ======================
# ======================================================================
st.divider()
st.subheader("4. Extract results from ODB")

@st.fragment
def render_extract_results_section(stl_dir: str, file_name: str) -> None:
    # if the user hasn't selected an STL file, show an error
    if "static" in st.session_state["sim_config"]["script_name"]:
        if not stl_dir or not file_name:
            st.error("Please select an STL file first.")
        if not st.session_state.get("sim_output_dir"):
            st.error("Simulation output directory not found. Please run the simulation first.")
        else:
            # ask user which field output to extract, and where to save it
            #field_name = st.selectbox("Select the field output to extract", ["U.Magnitude","U.U1", "U.U2", "U.U3"], index=0)
            FIELD_NAMES = ["U.Magnitude", "U.U1", "U.U2", "U.U3"]
            # Fixed per-field color (categorical palette, adjacent-pairlist order) so
            # a given field always gets the same hue, regardless of which fields end
            # up succeeding below.
            FIELD_COLORS = {
                "U.Magnitude": "#2a78d6",  # blue
                "U.U1": "#eb6834",         # orange
                "U.U2": "#1baf7a",         # aqua
                "U.U3": "#eda100",         # yellow
            }
            if st.button("Extract results"):
                nset_name = "SET-TOP"  # hard-coded in the ABAQUS simulation script
                # One shared figure: every field's boxplot becomes its own row on a
                # common value axis, so distributions are directly comparable.
                fig = go.Figure()
                for field_name in FIELD_NAMES:
                    # grab the simulation output folder from the session state, which was set by the create_simulation_section() function above.
                    sim_output_dir = st.session_state["sim_output_dir"]
                    # the ODB file is always named after the STL file, with a .odb extension
                    odb_path = sim_output_dir / f"run\\Job-{file_name}.odb"
                    # the output CSV file will be named after the STL file, with a suffix indicating the field output extracted.
                    out_path = sim_output_dir / f"run\\{file_name}_stress.csv"
                    success = abaqus_tools.extract_field_at_nodeset(str(odb_path), nset_name, field_name, str(out_path))
                    if success:
                        # ------ read the extracted values back and add them as a row in the shared boxplot ------
                        try:
                            results_df = pd.read_csv(out_path)
                        except Exception as e:
                            st.error(f"Could not read extracted results from {out_path}: {e}")
                        else:
                            fig.add_trace(go.Box(
                                x=results_df["value"], name=field_name, boxpoints="outliers",
                                orientation="h", marker_color=FIELD_COLORS[field_name],
                            ))
                    else:
                        st.error("Failed to extract results.")

                if fig.data:
                    fig.update_layout(
                        title=f"Field output distributions at node set '{nset_name}'",
                        xaxis_title="Value",
                        # Each row is already direct-labeled by field name, so a
                        # legend duplicating those labels would be redundant.
                        showlegend=False,
                    )
                    st.plotly_chart(fig)
    else:
        st.error(f"No export defined for this simulation type: {st.session_state['sim_config']['load_case']}. Please run a simulation with a load case first.")

if st.session_state.get("sim_output_dir") is not None:
    render_extract_results_section(stl_dir, file_name)
else:
    st.info("Run a simulation first to extract results from the ODB file.")

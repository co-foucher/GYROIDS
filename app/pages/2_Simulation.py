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

from gyroid_utils import TET_mesh_tools, abaqus_tools

from app.state import init_state, get_output_dir
from app.components.jobs import start_job, render_job_status
from app.components.Simulation_source import mesh_job, create_abaqus_job, run_abaqus_job
from app.components.file_picker import browse_file, browse_directory
from app.components.tpms_source_panel import load_STL
from app.components.mesh_preview import render_mesh_preview

st.set_page_config(page_title="Simulation", layout="wide")
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

    render_job_status(st.session_state["jobs"].get(st.session_state.get("mesh_job_id")))

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
    "Static analysis": "Perform a static analysis of the structure under a given load, in a given direction.",
}

st.subheader("2. create ABAQUS simulation")

col_1, col_2 = st.columns([1, 1.4])
with col_1:
    # -------  select the simulation type ------
    script_accro = st.selectbox("TPMS type", list(BUILTIN_SIM.keys()))
    script_name = BUILTIN_SIM[script_accro]

    # ------ give material properties for the simulation ------
    st.write("Material properties for the simulation :")
    st.caption("ABAQUS style - you have to make sure the units are consistent with the STL file's units.")
    young_modulus = st.number_input("Young's Modulus", value=300000.0, format="%.1f")
    poisson_ratio = st.number_input("Poisson's Ratio", value=0.21, format="%.2f")
    density = st.number_input("Density", value=3.9e-09, format="%.2e")
with col_2:
    st.write("Simulation type info :")
    st.caption(BUILTIN_SIM_INFO[script_accro])



if st.button("Create ABAQUS simulation input"):
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
                        young_modulus=young_modulus,
                        poisson_ratio=poisson_ratio,
                        density=density)
    st.session_state["abaqus_job_id"] = job_id

render_job_status(st.session_state["jobs"].get(st.session_state.get("abaqus_job_id")))


# ======================================================================
# ====================== run ABAQUS simulation =========================
# ======================================================================
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

render_job_status(st.session_state["jobs"].get(st.session_state.get("run_job_id")))
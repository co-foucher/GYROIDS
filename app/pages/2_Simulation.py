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

from gyroid_utils import TET_mesh_tools, abaqus_tools

from app.state import init_state, get_output_dir
from app.jobs import start_job
from app.components.job_log import render_job_status

st.set_page_config(page_title="Simulation", layout="wide")
init_state()
st.title("Mesh + Simulation")

default_dir = str(get_output_dir())

st.subheader("1. Tetrahedral meshing (fTetWild)")
stl_dir = st.text_input("Folder containing the .stl", value=default_dir)
file_name = st.text_input("File name (no extension)", value="my_tpms")
ftetwild_path = st.text_input(
    "fTetWild executable path",
    value=r"C:\Program Files\fTetWild\build\Release\FloatTetwild_bin.exe",
)
c1, c2 = st.columns(2)
epsilon = c1.number_input("Epsilon (envelope size)", value=0.001, format="%.5f")
cpu_cores = c2.number_input("CPU cores", value=1, min_value=1, step=1)

if st.button("Run fTetWild meshing"):
    def _mesh_job(log, stl_dir=stl_dir, file_name=file_name,
                 ftetwild_path=ftetwild_path, epsilon=epsilon, cpu_cores=cpu_cores):
        log(f"Meshing {file_name}.stl with fTetWild (this can take a while)...")
        TET_mesh_tools.mesh_an_STL(
            input_path=stl_dir + "/",
            output_path=stl_dir + "/",
            file_name=file_name,
            FtetWild_path=ftetwild_path,
            epsilon=epsilon,
            CPU_cores=int(cpu_cores),
        )
        log("fTetWild meshing finished - .inp file written.")
        return True

    job_id = start_job(st.session_state["jobs"], f"mesh:{file_name}", _mesh_job)
    st.session_state["mesh_job_id"] = job_id

render_job_status(st.session_state["jobs"].get(st.session_state.get("mesh_job_id")))

st.divider()

st.subheader("2. ABAQUS simulation")
script_name = st.text_input(
    "ABAQUS script name (must live in the folder above)",
    value="generate_frequency_sim.py",
)

if st.button("Create ABAQUS simulation input"):
    def _abaqus_job(log, stl_dir=stl_dir, file_name=file_name, script_name=script_name):
        log("Invoking ABAQUS (noGUI) to create the simulation input...")
        ok = abaqus_tools.create_simulation(
            input_path=stl_dir + "/",
            output_path=stl_dir + "/",
            file_name=file_name,
            script_name=script_name,
        )
        log(f"create_simulation() returned: {ok}")
        return ok

    job_id = start_job(st.session_state["jobs"], f"abaqus:{file_name}", _abaqus_job)
    st.session_state["abaqus_job_id"] = job_id

render_job_status(st.session_state["jobs"].get(st.session_state.get("abaqus_job_id")))

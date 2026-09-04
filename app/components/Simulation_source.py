"""Renders the live status/log of an app.jobs.Job."""
import streamlit as st
import time
from pathlib import Path
from typing import Callable, List, Optional
from gyroid_utils import TET_mesh_tools, abaqus_tools

from app.components.jobs import Job


"""
#=====================================================================================================================
0 - (reserved)
1 - render_job_status
#=====================================================================================================================
"""


# =====================================================================
# 1) mesh_job
# =====================================================================
def mesh_job(log: Callable[[str], None],
                stl_dir: str,
                file_name: str,
                ftetwild_path: str,
                epsilon: float,
                cpu_cores: int):
    """
    Runs fTetWild meshing in a background thread and stores its live status/log in st.session_state.
    
    PARAMETERS
    ----------
    log : callable
        Function to call with progress messages (e.g. `log("message")`).
    stl_dir : str
        Directory containing the STL file to mesh.
    file_name : str
        Name of the STL file (without extension) to mesh.
    ftetwild_path : str
        Path to the fTetWild executable.
    epsilon : float
        Envelope size for meshing.
    cpu_cores : int
        Number of CPU cores to use for meshing.
    RETURNS
    -------
    bool
        True if meshing was successful, False otherwise.          
    
    """
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



# =====================================================================
# 2) create_abaqus_job
# =====================================================================
def create_abaqus_job(log: Callable[[str], None],
                stl_dir: str,
                file_name: str,
                script_dir: str,
                script_name: str,
                young_modulus: float,
                poisson_ratio: float,
                density: float,
                quadratic_tets: int = 0,
                extra_args: Optional[dict] = None):
    """
    Runs the chosen Abaqus generator script (noGUI) to turn the tet mesh into
    a solver-ready Job-<file_name>.inp.

    `extra_args` carries the options that only some simulation types take -
    the static/stiffness case needs axis, load, compression and tol_frac,
    the frequency case needs none. They're forwarded as key=value arguments
    on the Abaqus command line; see abaqus_tools.create_simulation().
    """
    log("Invoking ABAQUS (noGUI) to create the simulation input...")
    if extra_args:
        log("Load case: " + ", ".join(f"{k}={v}" for k, v in extra_args.items()))
    ok = abaqus_tools.create_simulation(
        input_path=stl_dir + "/",
        output_path=str(script_dir),
        file_name=file_name,
        script_name=script_name,
        young_modulus=young_modulus,
        poisson_ratio=poisson_ratio,
        density=density,
        quadratic_tets=quadratic_tets,
        extra_args=extra_args,
    )
    log(f"create_simulation() returned: {ok}")
    if not ok:
        log(
            "Check generate_sim_logger_"
            f"{file_name}.txt in the simulation folder for the reason - for the "
            "static case it is usually an empty face node set (raise the face "
            "tolerance) or a mesh that has no flat ends along the load axis."
        )
    return ok


# =====================================================================
# 3a) _clear_previous_job_files
# =====================================================================
def _clear_previous_job_files(run_dir: Path, file_name: str, log: Callable[[str], None]) -> bool:
    """
    Deletes any leftover Job-<file_name>.* files (.odb, .sta, .msg, .lck,
    .log, ...) from a previous run in `run_dir`.

    WHY: when those files are already present, ABAQUS itself prompts
    "Old job files exist. Overwrite? (y/n):" on stdin before it will start.
    abaqus_tools.run_simulation() launches it via subprocess.run() with no
    stdin attached, so that prompt just hangs forever instead of failing or
    proceeding. Clearing old outputs before each run means there's nothing
    left for ABAQUS to ask about.

    RETURNS
    -------
    bool
        True if run_dir ended up clear of Job-<file_name>.* files. False if
        one or more couldn't be removed - typically the .odb, still locked
        by an earlier ABAQUS process (or a CAE/viewer session) that's still
        running against it.
    """
    run_dir = Path(run_dir)
    removed = []
    stuck = []
    for f in run_dir.glob(f"Job-{file_name}.*"):
        try:
            f.unlink()
            removed.append(f.name)
        except OSError as e:
            stuck.append(f.name)
            log(f"Could not remove old job file {f.name}: {e}")
    if removed:
        log(f"Removed leftover job files from a previous run: {', '.join(removed)}")
    if stuck:
        log(
            "The following files from a previous run are still locked and "
            f"could not be removed: {', '.join(stuck)}. This usually means an "
            "earlier ABAQUS process for this job is still running, or an "
            "ABAQUS CAE/viewer session still has the .odb open - close it "
            "and try again."
        )
    return not stuck



# =====================================================================
# 3) run_abaqus_job
# =====================================================================
def run_abaqus_job(log: Callable[[str], None],
                file_name: str,
                sim_dir: str,
                max_wait_time: int = 600):
    """
    Runs the ABAQUS solve for a simulation input already created by
    `create_abaqus_job`, then waits for it to complete.

    `sim_dir` is the "{file_name}_sim" folder that already holds
    Job-<file_name>.inp (written there by
    create_simulation/generate_frequency_sim.py). abaqus_tools.run_simulation
    copies that .inp from input_path to output_path before solving, so those
    two can't be the same path (shutil.copyfile raises SameFileError on a
    self-copy) - the actual solve therefore runs in a "run" subfolder of
    sim_dir, and that subfolder is what ends up holding the .odb/log files.

    PARAMETERS
    ----------
    log : callable
        Function to call with progress messages (e.g. `log("message")`).
    file_name : str
        Base name used to locate the input INP file and name the output files.
    sim_dir : str
        Folder holding Job-<file_name>.inp (written by create_abaqus_job).
    max_wait_time : int, optional
        Max seconds to wait (applied separately to the "job started" wait
        and the "job completed" wait). Default = 600.

    RETURNS
    -------
    bool
        True if the simulation started AND completed successfully, False otherwise.
    """
    sim_dir = Path(sim_dir)
    run_dir = sim_dir / "run"
    run_dir.mkdir(parents=True, exist_ok=True)

    # avoid ABAQUS's interactive "Old job files exist. Overwrite? (y/n):"
    # prompt, which would otherwise hang the background thread forever.
    cleared = _clear_previous_job_files(run_dir, file_name, log)
    if not cleared:
        log("Aborting - a previous job's files are still locked (see message above).")
        return False
    
    log(f"Starting ABAQUS job for {file_name}...")
    started = abaqus_tools.run_simulation(
        input_path=str(sim_dir),
        output_path=str(run_dir),
        file_name=file_name,
        max_wait_time=max_wait_time,
    )
    if not started:
        log("run_simulation() returned False - job did not start.")
        return False
    log("ABAQUS job started - waiting for it to complete...")

    completed = abaqus_tools.wait_for_simulation_completed(
        ODB_path=str(run_dir),
        file_name=file_name,
        max_wait_time=max_wait_time,
    )
    log(f"wait_for_simulation_completed() returned: {completed}")
    return completed


# =====================================================================
# 3) render_load_case_form
# =====================================================================
def render_load_case_form(key: str = "static") -> dict:
    """
    ============================================================================
    1) RENDER_LOAD_CASE_FORM
    Draws the inputs that define the uniaxial load case and returns them in
    the shape create_simulation(extra_args=...) expects.
    ============================================================================

    PARAMETERS
    ----------
    key : str, optional
        Unique suffix for the widget keys (default "static").

    RETURNS
    -------
    dict with keys
        "axis"        : "x" | "y" | "z"
        "load"        : float, total force over the loaded face
        "compression" : int, 1 = push, 0 = pull (int rather than bool so it
                        survives the trip through the Abaqus command line)
        "tol_frac"    : float, face slab thickness / axial extent

    NOTES
    -----
    The units are whatever the STL is in - the material properties entered
    above use the same convention, so with mm + MPa the load is in N.
    """
    axis = st.selectbox(
        "Load axis",
        ["z", "x", "y"],
        key=f"{key}_axis",
        help=(
            "The structure is gripped on its two extreme faces along this axis "
            "and loaded along it. Stiffness is measured in this direction."
        ),
    )
    c1, c2 = st.columns(2)
    load = c1.number_input(
        "Total load",
        value=100.0,
        format="%.4f",
        key=f"{key}_load",
        help=(
            "Total force spread over every node of the loaded face. Same unit "
            "system as the Young's modulus above (mm + MPa -> N)."
        ),
    )
    sense = c2.radio(
        "Direction",
        ["Compression", "Tension"],
        key=f"{key}_sense",
        horizontal=True,
        help="Compression pushes the top face into the structure; tension pulls it away.",
    )
    tol_pct = st.slider(
        "Face thickness (% of height)",
        min_value=0.1,
        max_value=10.0,
        value=1.0,
        step=0.1,
        key=f"{key}_tol",
        help=(
            "How thick a slab at each end counts as 'the face'. Too tight and "
            "the node set comes out empty and ABAQUS errors out; too loose and "
            "you end up clamping a real chunk of the structure. 1 % suits a "
            "flat-cut block; raise it if the ends are rough or wavy."
        ),
    )
    non_liner_geometry = st.toggle(
        "Non-linear geometry",
        help="Account for large deformations non-linearity.",
    )
    return {
        "axis": axis,
        "load": float(load),
        "compression": 1 if sense == "Compression" else 0,
        "tol_frac": tol_pct / 100.0,
        "non_linear_geometry": 1 if non_liner_geometry else 0,
    }
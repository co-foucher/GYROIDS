"""
Minimal background-job runner for long-running pipeline steps (fTetWild
meshing, ABAQUS simulation).

WHY THIS EXISTS
----------------
Streamlit reruns the entire script top-to-bottom on every widget
interaction and blocks the UI while a script run is in progress. The
simulation pipeline shells out to external, long-running processes
(fTetWild, ABAQUS - see gyroid_utils.TET_mesh_tools / abaqus_tools), so
calling them directly from a button handler would freeze the app for the
duration of the run. Instead, `start_job` runs the work in a background
thread and stores its live status/log in st.session_state.
"""
from __future__ import annotations

import threading
import traceback
import uuid
from dataclasses import dataclass, field
from typing import Callable, List, Optional
import streamlit as st


# =====================================================================
# 1) dataclass: Job
# =====================================================================
@dataclass
class Job:
    """
    Represents a background job with a status, log, and result.
    has an `append_log` method for adding progress messages.
    """
    id: str
    label: str
    status: str = "running"  # "running" | "done" | "error"
    log: List[str] = field(default_factory=list)    #is a list of strings, each string is a new log message
    result: object = None
    error: Optional[str] = None

    def append_log(self, line: str) -> None:
        self.log.append(line)

# =====================================================================
# 1) start_job
# =====================================================================
def start_job(jobs: dict, 
                label: str, 
                fn: Callable, 
                *args, 
                **kwargs) -> str:
    """
    Runs `fn(*args, log=job.append_log, **kwargs)` in a background thread.

    PARAMETERS
    ----------
    jobs : dict
        Registry to store the Job in - pass st.session_state["jobs"].
    label : str
        Human-readable label shown in the UI (e.g. "mesh:my_tpms").
    fn : callable, i.e. a function
        this function will be called in a background thread, 
        and it must accept a `log(str) -> None` keyword argument for progress messages.
        (see app/pages/2_Simulation.py for examples).
    *args : positional arguments
        all the argurments that need to be passed to `fn`.
        *args collects extra positional arguments — passed by position, no names, gathered into a tuple. 
    **kwargs : keyword arguments
        all the keyword arguments that need to be passed to `fn`.
        **kwargs collects extra keyword arguments — passed as name=value, gathered into a dict.
    
    NOTE:
    -----
    to undertand how *args and **kwargs work, consider the following example:

    def f(*args, **kwargs):
        print(args, kwargs)
    
    f(1, 2, x=3, y=4)
    # args = (1, 2)
    # kwargs = {'x': 3, 'y': 4}

    RETURNS
    -------
    job_id : str
        Key into `jobs` for polling status (see components/job_log.py).
    """
    # first create a Job object from the class defined above, with itsid defined by
    # a new UUID, and label defined by the label passed to the function. 
    job = Job(id=str(uuid.uuid4()), label=label)
    # add the job to the jobs dictionary (input of the function), 
    # with the job's id as the key and the job object as the value
    jobs[job.id] = job

    # define a runner function that will be executed in a background thread.
    def _runner():
        # call the provided function with the provided arguments, and pass the job's append_log method as the log argument.
        try:
            job.result = fn(*args, log=job.append_log, **kwargs)
            job.status = "done"
        # if an exception occurs, set the job's status to "error", and store the error message 
        # and traceback in the job's error attribute. Also append the error message to the job's log.
        except Exception as e:
            job.status = "error"
            job.error = f"{e}\n{traceback.format_exc()}"
            job.append_log(f"ERROR: {e}")

    # use the threading module to start a new thread that runs the _runner function.
    # dameon=True means that the thread will automatically exit when the main program exits.
    threading.Thread(target=_runner, daemon=True).start()
    #return the job's id so that the caller can use it to check the job's status and log.
    return job.id


# =====================================================================
# 3) render_job_status
# =====================================================================
def render_job_status(job: Job) -> None:
    """
    ============================================================================
    1) RENDER_JOB_STATUS
    Shows a job's current status and log tail. Streamlit doesn't poll
    background threads on its own, so a "Refresh status" button is offered
    click it (or add the `streamlit-autorefresh`
    package and call it once at the top of the page) to see progress update.
    ============================================================================

    PARAMETERS
    ----------
    job : app.jobs.Job or None
        The job to display. If None, shows an info message and returns
        immediately (e.g. before the user has started any job yet).

    RETURNS
    -------
    None
    """
    if job is None:
        st.info("No job has been started yet.")
        return

    if job.status == "running":
        st.info(f"Running: {job.label}")
    elif job.status == "done":
        # fn returning without raising just means the thread didn't crash -
        # it says nothing about whether the work itself succeeded. Jobs in
        # this app return a bool for that, so treat an explicit False result
        # as a (non-crashing) failure rather than showing it as success.
        if job.result is False:
            st.warning(f"Finished but reported failure: {job.label}")
        else:
            st.success(f"Done: {job.label}")
    else:
        st.error(f"Failed: {job.label}")
        if job.error:
            st.code(job.error)

    if job.log:
        st.code("\n".join(job.log[-200:]))

    if st.button("Refresh status", key=f"refresh_{job.id}"):
        st.rerun()
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
thread and stores its live status/log in st.session_state; the page polls
that state (see components/job_log.py) instead of blocking on the call.

This is intentionally simple (a thread + a dataclass), not a real task
queue - fine for a single local user driving one run at a time. If this
needs to survive a page reload/app restart, or run many jobs concurrently,
swap this for a proper job runner (e.g. a small SQLite-backed queue).
"""
from __future__ import annotations

import threading
import traceback
import uuid
from dataclasses import dataclass, field
from typing import Callable, List, Optional


@dataclass
class Job:
    id: str
    label: str
    status: str = "running"  # "running" | "done" | "error"
    log: List[str] = field(default_factory=list)
    result: object = None
    error: Optional[str] = None

    def append_log(self, line: str) -> None:
        self.log.append(line)


def start_job(jobs: dict, label: str, fn: Callable, *args, **kwargs) -> str:
    """
    Runs `fn(*args, log=job.append_log, **kwargs)` in a background thread.

    PARAMETERS
    ----------
    jobs : dict
        Registry to store the Job in - pass st.session_state["jobs"].
    label : str
        Human-readable label shown in the UI (e.g. "mesh:my_tpms").
    fn : callable
        Must accept a `log(str) -> None` keyword argument for progress
        messages; wrap existing gyroid_utils calls in a small local
        function rather than modifying them (see app/pages/2_Simulation.py
        for examples).

    RETURNS
    -------
    job_id : str
        Key into `jobs` for polling status (see components/job_log.py).
    """
    job = Job(id=str(uuid.uuid4()), label=label)
    jobs[job.id] = job

    def _runner():
        try:
            job.result = fn(*args, log=job.append_log, **kwargs)
            job.status = "done"
        except Exception as e:
            job.status = "error"
            job.error = f"{e}\n{traceback.format_exc()}"
            job.append_log(f"ERROR: {e}")

    threading.Thread(target=_runner, daemon=True).start()
    return job.id

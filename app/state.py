"""
Shared st.session_state helpers used across app/Home.py and app/pages/*.

Kept deliberately small: this is just the session-state wiring (current
model, output directory, background jobs registry), not pipeline logic -
that stays in gyroid_utils.
"""
from pathlib import Path

import streamlit as st

# Default folder for generated .npz/.stl/.html files. Overridable per
# session from the sidebar (see init_state()).
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent.parent / "gui_outputs"


def init_state() -> None:
    """
    Populates st.session_state with default keys the first time any page
    runs. Safe to call at the top of every page (setdefault is a no-op on
    reruns).
    """
    st.session_state.setdefault("output_dir", str(DEFAULT_OUTPUT_DIR))
    st.session_state.setdefault("current_model", None)      # last generated TPMSModel instance
    st.session_state.setdefault("current_equation", None)    # equation string, if a custom TPMS was used
    st.session_state.setdefault("jobs", {})                  # job_id -> app.jobs.Job

    with st.sidebar:
        st.session_state["output_dir"] = st.text_input(
            "Output folder",
            value=st.session_state["output_dir"],
            help="Where generated .stl/.html/.npz files are written and read from.",
        )


def get_output_dir() -> Path:
    """Returns the current session's output directory, creating it if needed."""
    d = Path(st.session_state.get("output_dir", DEFAULT_OUTPUT_DIR))
    d.mkdir(parents=True, exist_ok=True)
    return d

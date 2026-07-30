"""
Launches gyroid_utils.CT_visualization_window's interactive viewer as a
separate process for a given .mhd file already on disk. Shared by
app/pages/3_CT_Analysis.py (loaded volume) and app/components/ct_pipeline.py
(processed mask preview) so the subprocess/backend-forcing logic and the
full-vs-lightweight toggle only live in one place.

Two viewer modes, both from CT_visualization_window.py:
- Full (open_window): brush/rainbow/histogram tooling, scroll-to-scrub.
- Lightweight (lightweigth_open): just a Z-slice slider + click-to-inspect
  greyvalues - opens faster and uses less memory, handy for a quick look
  or for large volumes where the full toolset isn't needed.

Split into a toggle (render_lightweight_toggle) and a launcher
(launch_ct_viewer) rather than one combined button, so callers that need
to do work before launching (e.g. ct_pipeline.py writing a temp .mhd for
the current result) can keep that work gated behind their own "open
viewer" button click instead of it re-running on every page rerun.
"""
import os
import subprocess
import sys

import streamlit as st

__all__ = ["render_lightweight_toggle", "launch_ct_viewer"]


# =====================================================================
# 1) render_lightweight_toggle
# =====================================================================
def render_lightweight_toggle(key: str) -> bool:
    """Renders the "use lightweight viewer" checkbox and returns its state."""
    return st.checkbox(
        "Use lightweight viewer (slider only - faster, less memory)",
        value=False,
        key=f"{key}_lightweight",
    )


# =====================================================================
# 2) launch_ct_viewer
# =====================================================================
def launch_ct_viewer(mhd_path: str, lightweight: bool = False) -> None:
    """
    Launches gyroid_utils.CT_visualization_window on `mhd_path` in a
    separate process - open_window (full) or lightweigth_open, depending
    on `lightweight`.
    """
    # CT_visualization_window.py is written for Jupyter, where the user
    # runs `%matplotlib qt` themselves before calling it - it never sets
    # a backend on its own. Here there's no such magic, so matplotlib
    # falls back to whatever it auto-detects; if Streamlit's own process
    # has MPLBACKEND=Agg set (it often does, being headless-by-default),
    # a plain subprocess would inherit that env var and get the
    # non-interactive Agg backend ("FigureCanvasAgg is non-interactive"
    # warning, no window ever appears). Force QtAgg explicitly, and
    # strip any inherited MPLBACKEND so it can't override that.
    env = os.environ.copy()
    env.pop("MPLBACKEND", None)
    entrypoint = "lightweigth_open" if lightweight else "open_window"
    subprocess.Popen(
        [
            sys.executable, "-c",
            "import matplotlib; matplotlib.use('QtAgg'); "
            "import SimpleITK as sitk, gyroid_utils.CT_visualization_window as w; "
            f"w.{entrypoint}(sitk.ReadImage(r'{mhd_path}'))",
        ],
        env=env,
    )
    st.info(
        "Launching in a separate window - requires a local display "
        "and PyQt5/PySide installed (see CT_visualization_window.py)."
    )

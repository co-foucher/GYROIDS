"""
GYROIDS GUI - entry point.

Run with (from the repo root, after `pip install -e ".[gui]"`):
    streamlit run app/Home.py

This is a thin front end over the gyroid_utils library (src/gyroid_utils):
it implements no pipeline logic itself, only forms/wiring around the
existing TPMS / mesh / simulation / CT functions. See each page for its
current status.
"""
# Repo root isn't on sys.path by default (app/ is a sibling of src/, not
# part of the installable package) - add it before importing anything
# under `app.*`. See app/_bootstrap.py for why this can't be a shared
# import instead.
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st

from app.state import init_state

st.set_page_config(page_title="GYROIDS", page_icon=":ice_cube:", layout="wide")
init_state()

st.title("GYROIDS")
st.write(
    "GUI front end for the gyroid_utils pipeline. Use the pages in the "
    "sidebar for each stage of the workflow."
)
st.markdown(
    "- **Generate TPMS** - built-in surfaces or a custom equation, live preview, export STL\n"
    "- **Simulation** - mesh an STL with fTetWild and launch ABAQUS batches\n"
    "- **CT Analysis** - convert/inspect CT volumes\n"
    "- **Library** - browse previously generated structures"
)
st.info(
    "First-pass scaffold: **Generate TPMS** is fully functional. "
    "**Simulation** and **CT Analysis** wire up the real gyroid_utils calls "
    "but have minimal parameter coverage - extend as needed. **Library** is "
    "a simple file browser over the output folder."
)

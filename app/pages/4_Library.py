"""
Library: browse previously generated TPMS structures (.stl + .html preview
pairs) saved in the current output folder.

STATUS: scaffold - a plain file browser, no gyroid_utils calls needed.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st
import streamlit.components.v1 as components

from app.state import init_state, get_output_dir

st.set_page_config(page_title="Library", layout="wide")
init_state()
st.title("Library")

out_dir = get_output_dir()
st.caption(f"Scanning: {out_dir}")

stl_files = sorted(out_dir.glob("*.stl"))

if not stl_files:
    st.info("No exported STL files found yet. Generate one on the 'Generate TPMS' page.")
else:
    for stl_path in stl_files:
        html_path = stl_path.with_suffix(".html")
        with st.expander(stl_path.name):
            col1, col2 = st.columns([1, 2])
            with col1:
                st.write(f"`{stl_path.name}`")
                st.download_button(
                    "Download STL",
                    data=stl_path.read_bytes(),
                    file_name=stl_path.name,
                    key=f"dl_{stl_path.name}",
                )
            with col2:
                if html_path.exists():
                    components.html(html_path.read_text(encoding="utf-8"), height=400, scrolling=True)
                else:
                    st.caption("No preview .html saved alongside this STL.")

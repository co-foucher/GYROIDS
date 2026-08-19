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
        # Naming convention set by the "Export ..." buttons on the Generate
        # TPMS page: the field is saved as "<name>.npy" and the thickness
        # field (if exported separately) as "<name>_thickness.npy", both
        # sharing the STL's base name.
        field_npy_path = stl_path.with_suffix(".npy")
        thickness_npy_path = stl_path.with_name(stl_path.stem + "_thickness.npy")


        with st.expander(f"{stl_path.name}"):
            col1, col2 = st.columns([1, 2])
            with col1:
                st.success(f"`{stl_path.name}`")
                st.download_button(
                    "Download STL",
                    data=stl_path.read_bytes(),
                    file_name=stl_path.name,
                    key=f"dl_{stl_path.name}",
                )

                if field_npy_path.exists():
                    st.success(f"Field available: `{field_npy_path.name}`")
                    st.download_button(
                        "Download field .npy",
                        data=field_npy_path.read_bytes(),
                        file_name=field_npy_path.name,
                        key=f"dl_field_npy_{stl_path.name}",
                    )
                else:
                    st.warning("No field .npy saved alongside this STL.")

                if thickness_npy_path.exists():
                    st.success(f"Thickness field available: `{thickness_npy_path.name}`")
                    st.download_button(
                        "Download thickness .npy",
                        data=thickness_npy_path.read_bytes(),
                        file_name=thickness_npy_path.name,
                        key=f"dl_thickness_npy_{stl_path.name}",
                    )
                else:
                    st.warning("No thickness .npy saved alongside this STL.")
            with col2:
                if html_path.exists():
                    components.html(html_path.read_text(encoding="utf-8"), height=400, scrolling=True)
                else:
                    st.caption("No preview .html saved alongside this STL.")

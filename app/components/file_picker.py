"""
Native OS file/folder dialogs for local desktop use, via tkinter.

Streamlit has no built-in file picker that returns a real filesystem path -
st.file_uploader() only hands you the file's bytes, which doesn't work well
for a .mhd volume (the actual voxel data lives in a sibling .raw/.zraw file
found via a relative path inside the .mhd header; uploading just the .mhd
bytes leaves that reference dangling) or for a folder of JPG/DICOM/TIFF
slices (there could be thousands of files). Since this app runs as a local
Streamlit server on the user's own machine (see app/_bootstrap.py and the
existing "Open interactive CT viewer" button, which already assumes a local
display), popping a native OS dialog from the server process works fine -
it appears on the same screen as the browser tab.
"""
from typing import Callable, List, Optional, Tuple

import streamlit as st

__all__ = ["browse_file", "browse_directory"]


# =====================================================================
# 0 - (reserved)
# 1 - _run_native_dialog
# 2 - _render_browse_button
# 3 - browse_file
# 4 - browse_directory
# =====================================================================

# =====================================================================
# 1) _run_native_dialog
# =====================================================================
def _run_native_dialog(key: str, method: str, **dialog_kwargs) -> None:
    """
    Shared plumbing for browse_file/browse_directory: opens a hidden,
    topmost Tk root, calls the named tkinter.filedialog method on it, tears
    the root down, and writes the result into st.session_state[key] (or an
    error message into st.session_state[f"{key}_browse_error"] if tkinter
    isn't available or the dialog fails - e.g. no local display).

    PARAMETERS
    ----------
    key : str
        The st.session_state key to write the chosen path into.
    method : str
        Name of the tkinter.filedialog function to call, e.g.
        "askopenfilename" or "askdirectory".
    **dialog_kwargs
        Forwarded to that function (title, filetypes, ...).
    """
    try:
        import tkinter as tk
        from tkinter import filedialog
    except ImportError:
        st.session_state[f"{key}_browse_error"] = (
            "tkinter is not available in this Python environment - can't "
            "open a native file dialog. Type/paste the path instead."
        )
        return

    try:
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        try:
            path = getattr(filedialog, method)(**dialog_kwargs)
        finally:
            root.destroy()
    except Exception as e:
        st.session_state[f"{key}_browse_error"] = f"Could not open dialog: {e}"
        return

    st.session_state[f"{key}_browse_error"] = None
    if path:
        st.session_state[key] = path


# =====================================================================
# 2) _render_browse_button
# =====================================================================
def _render_browse_button(key: str, on_click: Callable[[], None]) -> None:
    """Renders the "Browse..." button + any error left by the last click."""
    st.button("Browse...", key=f"{key}_browse_btn", on_click=on_click)
    error = st.session_state.get(f"{key}_browse_error")
    if error:
        st.error(error)


# =====================================================================
# 3) browse_file
# =====================================================================
def browse_file(key: str, title: str = "Select a file",
                filetypes: Optional[List[Tuple[str, str]]] = None) -> None:
    """
    Renders a "Browse..." button that opens a native OS file-open dialog
    and writes the chosen path into st.session_state[key]. Pair it with a
    st.text_input(..., key=key) rendered alongside it, so the user can see
    the result and still edit/paste a path by hand - see
    app/pages/3_CT_Analysis.py for the pattern.

    PARAMETERS
    ----------
    key : str
        The st.session_state key to write the chosen path into. Must
        already exist in st.session_state (e.g. via
        st.session_state.setdefault(key, "")) before the paired
        text_input is created.
    title : str, optional
        Dialog window title.
    filetypes : list of (label, pattern) tuples, optional
        Passed straight to tkinter's filedialog, e.g.
        [("MHD files", "*.mhd"), ("All files", "*.*")]. Defaults to
        "All files" only.

    RETURNS
    -------
    None. Writes to st.session_state[key] as a side effect of the button's
    on_click callback (runs before the page reruns, so the paired
    text_input picks up the new value on the same rerun - the standard
    Streamlit pattern for populating a widget from a callback).

    NOTES
    -----
    Requires a local display and tkinter (bundled with standard Python
    installs on Windows/macOS; on Linux may need the `python3-tk` system
    package). If unavailable, shows an st.error instead of crashing the
    whole page.
    """
    _render_browse_button(key, lambda: _run_native_dialog(
        key, "askopenfilename",
        title=title, filetypes=filetypes or [("All files", "*.*")],
    ))


# =====================================================================
# 4) browse_directory
# =====================================================================
def browse_directory(key: str, title: str = "Select a folder") -> None:
    """
    Same as browse_file(), but opens a native OS folder-select dialog
    (tkinter's askdirectory) instead of a file-open dialog - for inputs
    like a folder of JPG/DICOM/TIFF slices, where convert_*_to_mhd()
    accepts a directory path directly.

    PARAMETERS
    ----------
    key : str
        The st.session_state key to write the chosen folder path into.
        Must already exist in st.session_state before the paired
        text_input is created.
    title : str, optional
        Dialog window title.

    RETURNS
    -------
    None. See browse_file() for the callback/session_state details.
    """
    _render_browse_button(key, lambda: _run_native_dialog(
        key, "askdirectory", title=title,
    ))

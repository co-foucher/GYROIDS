import numpy as np
import plotly.graph_objects as go
import sympy as sp
import streamlit as st

# =====================================================================
# 0 - (reserved)
# 1 - import_matrix_from_file
# =====================================================================

# =====================================================================
# 1) import_matrix_from_file
# =====================================================================
def import_matrix_from_file(file_path):
    """
    ============================================================================
    1) IMPORT_MATRIX_FROM_FILE
    Imports a matrix from a numpy or csv file.
    ============================================================================

    PARAMETERS
    ----------
    file_path : str
        Path to the file containing the matrix (.npy, or a text file
        readable by np.loadtxt).

    RETURNS
    -------
    matrix : np.ndarray or None
        The imported matrix, or None if import failed (also shows an
        st.error with the exception message).
    """
    try:
        if file_path.endswith(".npy"):
            matrix = np.load(file_path)
        else:
            matrix = np.loadtxt(file_path)
        return matrix
    except Exception as e:
        st.error(f"Error importing matrix from file: {e}")
        return None
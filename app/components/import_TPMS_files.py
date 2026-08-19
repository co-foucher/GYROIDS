import numpy as np
import plotly.graph_objects as go
import sympy as sp
import streamlit as st

# =====================================================================
# 0 - (reserved)
# 1 - import_matrix_from_file
# =====================================================================

def import_matrix_from_file(file_path):
    """
    Imports a matrix from a numpy or csv file. 
    
    Parameters:
    - file_path: str, path to the text file containing the matrix.
    
    Returns:
    - np.ndarray: The imported matrix as a NumPy array.
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
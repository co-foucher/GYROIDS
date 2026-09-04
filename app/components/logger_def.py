
import streamlit as st
from gyroid_utils.logger import logger

@st.fragment
def set_log_level():
    """
    Set the log level for the logger based on the user's selection in the Streamlit app.
    """
    log_level = st.selectbox(
        "Select Log Level",
        options=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        index=1,  # Default to INFO
        key="log_level_selector"
    )
    logger.setLevel(log_level)
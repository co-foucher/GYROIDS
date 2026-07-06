import logging
import sys


# Create top-level logger
logger = logging.getLogger("gyroid_utils")
logger.setLevel(logging.INFO)  # default level

# Create handler (stdout)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)

# Nice formatting
formatter = logging.Formatter(
    "[%(levelname)s] %(name)s:((%(funcName)s)): %(message)s"
)
handler.setFormatter(formatter)

# Avoid adding multiple handlers if re-imported
if not logger.handlers:
    logger.addHandler(handler)


"""
#=====================================================================================================================
0 - (reserved)
1 - set_log_level
#=====================================================================================================================
"""


# =====================================================================
# 1) set_log_level
# =====================================================================
def set_log_level(level: str):
    """
    ============================================================================
    1) SET_LOG_LEVEL
    Sets the global logging level for the gyroid_utils package.
    ============================================================================

    PARAMETERS
    ----------
    level : str
        One of: 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'.

    RETURNS
    -------
    None

    EXAMPLE
    -------
    >>> import gyroid_utils
    >>> gyroid_utils.set_log_level("DEBUG")
    """
    logger.setLevel(level.upper())

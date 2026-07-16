"""
Custom TPMS equation support: parsing/validation/evaluation of user-typed
formula strings, the Streamlit widgets that collect them, and the one call
site (evaluate_custom_inputs) that turns them into the plain numpy arrays
gyroid_utils.TPMS_classes.tpms_custom.CustomTPMSModel expects. Used by
app/pages/1_Generate_TPMS.py.

WHY THIS LIVES HERE, NOT IN THE LIBRARY
-----------------------------------------
CustomTPMSModel takes a precomputed `field` array (and a precomputed
`thickness` array/scalar) - plain numpy data, exactly like every other
TPMSModel subclass. It has no idea a user might have typed a formula
string into a text box; that's a GUI concern, not a library one.
Everything about turning that string into numbers - parsing, validating,
sympy, the max/min duck-typing behavior described below - lives here
instead, so the library stays a plain numerical package that only ever
sees numpy arrays. 1_Generate_TPMS.py doesn't need to know equations are
involved at all - it just calls render_equation_input() for the widgets
and evaluate_custom_inputs() to get arrays back.
"""
from typing import Callable, Optional, Tuple

import numpy as np
import plotly.graph_objects as go
import sympy as sp
import streamlit as st

__all__ = ["render_equation_input", "evaluate_custom_inputs", "EquationError"]


# =====================================================================
# 0 - (reserved)
# 1 - EquationError (exception)
# 2 - parse_equation
# 3 - validate_equation
# 4 - evaluate_equation
# 5 - render_equation_input
# 6 - evaluate_custom_inputs
# =====================================================================

# Only these symbols may appear in a user equation. No px/py/pz: for a
# custom equation, the user's formula *is* the whole field definition -
# there's nothing left for a separate period to parameterize (see
# CustomTPMSModel's docstring in the library for the same point from the
# library side).
_ALLOWED_SYMBOLS = ("x", "y", "z")


# =====================================================================
# max/min: same dual behavior as Python's own builtins
# =====================================================================
# Plain sympy Max/Min always compare N pointwise arguments elementwise
# (Max(x, y) -> the larger of the two arrays, elementwise).
# Thus, "max(abs(x))" # would silently evaluate to abs(x) unchanged, 
# not "the largest |x| over the whole domain".
# This fixes it
class _ReduceOrCompareMax(sp.Function):
    @classmethod
    def eval(cls, *args):
        if len(args) > 1:
            return sp.Max(*args)
        return None  # single arg: stay unevaluated -> reduction at eval time


class _ReduceOrCompareMin(sp.Function):
    @classmethod
    def eval(cls, *args):
        if len(args) > 1:
            return sp.Min(*args)
        return None


# Passed as the first entry of lambdify's `modules` list so a leftover,
# unevaluated single-argument _ReduceOrCompareMax/Min prints as a call to
# numpy's full-array reduction instead of sympy's elementwise one. Falls
# through to the second entry ("numpy") for every other allowed function.
_LAMBDIFY_MODULES = [
    {"_ReduceOrCompareMax": np.max, "_ReduceOrCompareMin": np.min},
    "numpy",
]

# Only these functions may appear in a user equation. Deliberately a small,
# conservative set - extend if a use case needs more.
_ALLOWED_FUNCS = {
    "sin": sp.sin, "cos": sp.cos, "tan": sp.tan,
    "sinh": sp.sinh, "cosh": sp.cosh, "tanh": sp.tanh,
    "exp": sp.exp, "log": sp.log, "sqrt": sp.sqrt,
    "abs": sp.Abs, "Abs": sp.Abs,
    "max": _ReduceOrCompareMax, "min": _ReduceOrCompareMin,
    "pi": sp.pi,
}


# =====================================================================
# 1) EquationError
# =====================================================================
class EquationError(ValueError):
    """Raised when a user-supplied equation string is invalid."""


# =====================================================================
# 2) parse_equation
# =====================================================================
def parse_equation(equation: str) -> Tuple[sp.Expr, Callable]:
    """
    ============================================================================
    2) PARSE_EQUATION
    Parses a formula string into a validated sympy expression and a fast
    numpy-vectorized callable fn(x, y, z) -> ndarray.
    ============================================================================

    PARAMETERS
    ----------
    equation : str
        Implicit surface formula, e.g.
        "sin(pi*x/2)*cos(pi*y/2) + sin(pi*y/2)*cos(pi*z/2)
        + sin(pi*z/2)*cos(pi*x/2)".
        Allowed variables: x, y, z.
        Allowed functions: sin, cos, tan, sinh, cosh, tanh, exp, log, sqrt,
        abs/Abs, max/min (see the module-level note above on their
        Python-like, argument-count-dependent behavior), plus the
        constant pi.

    RETURNS
    -------
    expr : sympy.Expr
        The parsed symbolic expression (useful for display/debugging).
    fn : callable
        fn(x, y, z) -> ndarray, vectorized with numpy.

    RAISES
    ------
    EquationError
        If the string can't be parsed, or references a symbol/function
        outside the allowed set.
    """
    symbols = {name: sp.Symbol(name) for name in _ALLOWED_SYMBOLS}
    namespace = {**_ALLOWED_FUNCS, **symbols}

    try:
        expr = sp.sympify(equation, locals=namespace)
    except (sp.SympifyError, TypeError, SyntaxError, AttributeError) as e:
        raise EquationError(f"Could not parse equation '{equation}': {e}") from e

    unknown = expr.free_symbols - set(symbols.values())
    if unknown:
        names = ", ".join(sorted(str(s) for s in unknown))
        raise EquationError(
            f"Equation uses unknown symbol(s): {names}. "
            f"Allowed variables are: {', '.join(_ALLOWED_SYMBOLS)}."
        )

    try:
        fn = sp.lambdify([symbols[n] for n in _ALLOWED_SYMBOLS], expr, modules=_LAMBDIFY_MODULES)
    except Exception as e:
        raise EquationError(f"Could not compile equation '{equation}': {e}") from e

    return expr, fn


# =====================================================================
# 3) validate_equation
# =====================================================================
def validate_equation(equation: str) -> Tuple[bool, str]:
    """
    ============================================================================
    3) VALIDATE_EQUATION
    Validates a formula string without needing a full-resolution coordinate
    grid - parses it, then smoke-tests it on a tiny 4x4x4 grid to catch
    runtime issues sympy's parser can't (e.g. log of a negative number,
    shape mismatches). Intended for a GUI "validate as you type" widget.
    ============================================================================

    PARAMETERS
    ----------
    equation : str
        The formula string to validate.

    RETURNS
    -------
    ok : bool
        True if the equation parsed and evaluated cleanly.
    message : str
        Human-readable explanation (error detail, or the parsed form on
        success).

    EXAMPLE
    -------
    >>> validate_equation("sin(x) + cos(y)")
    (True, 'OK - parsed as: sin(x) + cos(y)')
    >>> validate_equation("sin(x) + banana(y)")
    (False, "Could not parse equation ...")
    """
    try:
        expr, fn = parse_equation(equation)
    except EquationError as e:
        return False, str(e)

    try:
        g = np.linspace(-1.0, 1.0, 4)
        gx, gy, gz = np.meshgrid(g, g, g, indexing="ij")
        result = np.asarray(fn(gx, gy, gz), dtype=float)
        result = np.broadcast_to(result, gx.shape)
        if not np.all(np.isfinite(result)):
            return False, "Equation produced non-finite values (inf/nan) on a test grid."
    except Exception as e:
        return False, f"Equation failed on test evaluation: {e}"

    # cosmetic only: show the user-facing names they typed rather than the
    # internal _ReduceOrCompareMax/Min class names used to get max/min's
    # argument-count-dependent behavior (see the module-level note above).
    display = sp.sstr(expr).replace("_ReduceOrCompareMax", "max").replace("_ReduceOrCompareMin", "min")
    return True, f"OK - parsed as: {display}"


# =====================================================================
# 4) evaluate_equation
# =====================================================================
def evaluate_equation(equation: str, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    """
    ============================================================================
    4) EVALUATE_EQUATION
    Parses and evaluates a formula on a real coordinate grid, broadcasting
    scalars/reductions (e.g. a single-argument max(...)) up to the grid
    shape. This is the one call site the GUI needs: it turns a raw string
    into the plain numpy array CustomTPMSModel expects.
    ============================================================================

    PARAMETERS
    ----------
    equation : str
        The formula string (see parse_equation for allowed syntax).
    x, y, z : np.ndarray
        Coordinate grids of identical shape.

    RETURNS
    -------
    field : np.ndarray
        The evaluated formula, broadcast to x's shape.

    RAISES
    ------
    EquationError
        If the formula is invalid or fails to evaluate on this grid.

    EXAMPLE
    -------
    >>> field = evaluate_equation("sin(pi*x/2) + cos(pi*y/2)", x, y, z)
    """
    _, fn = parse_equation(equation)
    try:
        result = fn(x, y, z)
    except Exception as e:
        raise EquationError(f"Equation '{equation}' failed to evaluate: {e}") from e

    return np.broadcast_to(np.asarray(result, dtype=float), np.asarray(x).shape).copy()


DEFAULT_EQUATION = (
    "sin(pi * x / 2) * cos(pi * y / 2) + "
    "sin(pi * y / 2) * cos(pi * z / 2) + "
    "sin(pi * z / 2) * cos(pi * x / 2) "
)


# =====================================================================
# 5) render_equation_input
# =====================================================================
def render_equation_input(key_prefix: str = "eq") -> Optional[str]:
    """
    Renders the formula text box, validates it, and (if valid) shows a
    cheap 2D mid-slice preview computed on a small grid - independent of
    the full-resolution grid used for actual generation, so feedback is
    near-instant even before clicking "Generate".

    RETURNS
    -------
    equation : str or None
        The equation string if it validated successfully, else None (the
        caller should disable the Generate button while this is None).
    """
    st.caption("Variables: x, y, z, pi ")
    st.caption("functions: sin, cos, tan, sinh, cosh, tanh, exp, log, sqrt, abs, max, min")

    # =================================================
    # ============ define implicit field ==============
    # =================================================
    equation = st.text_area(
        "Implicit surface equation F(x, y, z)",
        value=st.session_state.get(f"{key_prefix}_text", DEFAULT_EQUATION),
        key=f"{key_prefix}_text",
        height=80,
    )
    ok_eq, message = validate_equation(equation)
    if not ok_eq:
        st.error(message)
        return None
    st.success(message)

    # =================================================
    # =========== visualize implicit field ============
    # =================================================

    with st.expander("Preview of Field (mid z-slice, low-res, sample from -2 to 2)", expanded=False):
        _, fn = parse_equation(equation)
        n = 120
        g = np.linspace(-2.0, 2.0, n)
        gx, gy = np.meshgrid(g, g, indexing="ij")
        gz = np.zeros_like(gx)
        field = fn(gx, gy, gz)
        field = np.broadcast_to(np.asarray(field, dtype=float), gx.shape)

        fig = go.Figure(go.Heatmap(x=g, y=g, z=field.T, colorscale="Portland"))
        fig.update_layout(height=350, margin=dict(l=0, r=0, t=20, b=0))
        st.plotly_chart(fig, use_container_width=True)

    # =================================================
    # ======== define thickness of surface ============
    # =================================================
    thickness = st.text_area(
        "Thickness",
        value=st.session_state.get(f"{key_prefix}_thickness", "0.2 + 0.3 * x / max(abs(x))"),
        key=f"{key_prefix}_thickness",
        height=80,
    )
    ok_th, message = validate_equation(thickness)
    if not ok_th:
        st.error(message)
        return None
    st.success(message)

    return equation, thickness


# =====================================================================
# 6) evaluate_custom_inputs
# =====================================================================
def evaluate_custom_inputs(equation: str, thickness: str,
                           x: np.ndarray, y: np.ndarray, z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Turns the (equation, thickness) strings captured by
    render_equation_input() into the plain numpy arrays CustomTPMSModel
    expects, evaluated on the real generation grid. This is the only place
    that should call evaluate_equation() for the "Generate" flow - keeps
    1_Generate_TPMS.py itself free of any parsing-related imports.

    PARAMETERS
    ----------
    equation, thickness : str
        The strings returned by render_equation_input().
    x, y, z : np.ndarray
        The real (full-resolution) coordinate grids to evaluate on - not
        the small preview grid used inside render_equation_input().

    RETURNS
    -------
    field, thickness_value : np.ndarray, np.ndarray
        Evaluated arrays, each the same shape as x.

    RAISES
    ------
    EquationError
        If either formula fails to evaluate on this grid. This can happen
        even though render_equation_input() already validated both
        strings: that validation only smoke-tests on a tiny grid, and
        formulas involving log/sqrt/etc. can behave differently over the
        real domain.
    """
    field = evaluate_equation(equation, x, y, z)
    thickness_value = evaluate_equation(thickness, x, y, z)
    return field, thickness_value

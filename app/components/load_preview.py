"""
Streamlit widgets for the uniaxial (stiffness) load case: the parameter
form and the 3D preview of where the load and the supports land.

Mirrors app/components/mesh_preview.py's split - a cached, st.*-free
figure builder plus a thin rendering function - because widgets can't live
inside an @st.cache_data function.
"""
import numpy as np
import plotly.graph_objects as go
import streamlit as st

from gyroid_utils import viz
from gyroid_utils.logger import logger


"""
#=====================================================================================================================
0 - (reserved)
1 - (moved to Simulation_source.render_load_case_form)
2 - _build_load_figure
3 - render_load_preview
4 - _sample_face_points
5 - build_load_preview_figure
6 - _axis_index
7 - select_end_faces
8 - pick_rbm_nodes
#=====================================================================================================================
"""


# =====================================================================
# 2) _build_load_figure
# =====================================================================
@st.cache_data(show_spinner="Building load preview...", max_entries=8)
def _build_load_figure(mesh_name: str,
                        _faces, 
                        _verts, 
                        axis: str, 
                        _load: float,
                        compression: bool, 
                        tol_frac: float,
                        height: int = 600) :
    """
    Pure, cacheable half of render_load_preview(). Deliberately free of any
    st.* call - see mesh_preview._build_mesh_figure() for why the two halves
    are kept apart.
    """
    fig, rmb = build_load_preview_figure(
        _faces, 
        _verts, 
        axis=axis, 
        load=_load,
        compression=compression, 
        tol_frac=tol_frac,
    )
    fig.update_layout(height=height)
    return fig, rmb


# =====================================================================
# 3) render_load_preview
# =====================================================================
def render_load_preview(mesh_name: str,
                        faces: np.ndarray, 
                        verts: np.ndarray, 
                        case: dict, 
                        key: str = "static",
                        height: int = 600) -> None:
    """
    ============================================================================
    3) RENDER_LOAD_PREVIEW
    Embeds the load-case preview: the STL with red arrows on the loaded face
    and support symbols on the held face, so the user can check the load is
    where they think it is before starting a long solve.
    ============================================================================

    PARAMETERS
    ----------
    faces, verts : ndarray or None
        Mesh data (as returned by tpms_source_panel.load_STL()). If either
        is None, a placeholder is shown instead.
    case : dict
        Load case, as returned by render_load_case_form().
    key : str, optional
        Unique suffix for widget keys (default "static").
    height : int, optional
        Plotly figure height in px (default 600).

    RETURNS
    -------
    rbm : (2, 3) ndarray
        The cooridnates of the two points on the bottom face that remove the rigid-body modes
        left free by the roller support.
    """
    if faces is None or verts is None:
        st.info("Select an STL first to preview the load case.")
        return

    try:
        fig, rbm = _build_load_figure(mesh_name=mesh_name,
            _faces = faces, 
            _verts =verts,
            axis=case["axis"],
            _load=case["load"],
            compression=bool(case["compression"]),
            tol_frac=case["tol_frac"],
            height=height,
        )
    except ValueError as e:
        st.error(f"Cannot build the load case: {e}")
        return

    #
    st.plotly_chart(fig, width="stretch", key=f"{key}_loadfig")
    st.caption(
        "Red arrows: nodes carrying the load (total load split equally between "
        "them). Blue crosses: the roller face, held only along the load axis so "
        "the structure can still expand sideways. Orange diamonds: the two "
        "in-plane pins that remove the leftover rigid-body motion."
    )
    return rbm



# =====================================================================
# 4) _sample_face_points
# =====================================================================
def _sample_face_points(pts: np.ndarray, t1: int, t2: int, n_target: int = 48) -> np.ndarray:
    """
    gives a face's point cloud down to ~n_target points that are spread
    over the face instead of clustered wherever the STL happens to be
    finely tessellated.
    thank you AI for this function ><
    """
    if len(pts) <= n_target:
        return pts

    n_side = max(2, int(np.ceil(np.sqrt(n_target))))

    def _bin(v, n):
        lo, hi = v.min(), v.max()
        if hi <= lo:
            return np.zeros(len(v), dtype=int)
        return np.clip(((v - lo) / (hi - lo) * n).astype(int), 0, n - 1)

    key = _bin(pts[:, t1], n_side) * n_side + _bin(pts[:, t2], n_side)
    _, first = np.unique(key, return_index=True)
    return pts[first]


# =====================================================================
# 5) build_load_preview_figure
# =====================================================================
def build_load_preview_figure(faces: np.ndarray,
                              verts: np.ndarray,
                              axis="z",
                              load: float = 1.0,
                              compression: bool = True,
                              tol_frac: float = 0.01,
                              n_arrows: int = 48,
                              height: int = 600):
    """
    ============================================================================
    5) BUILD_LOAD_PREVIEW_FIGURE
    Builds the "how is my load applied" figure: the STL surface, plus load
    arrows on the top face and support symbols on the bottom face, exactly
    where generate_static_sim.py will put the CLOAD and the boundary
    conditions.

    Arrows point *into* the structure for compression and *away* from it
    for tension, so the sign of the load case is readable at a glance.
    ============================================================================

    PARAMETERS
    ----------
    faces : (M, 3) ndarray
        Triangle indices.
    verts : (N, 3) ndarray
        Vertex coordinates.
    axis : str or int, optional
        Load axis (default "z") - see axis_index().
    load : float, optional
        Total load magnitude, only used for the annotation/hover text
        (default 1.0). Units follow the STL - see the GUI's caption.
    compression : bool, optional
        True (default) = the top face is pushed down onto the structure;
        False = pulled away from it.
    tol_frac : float, optional
        Face slab thickness as a fraction of the axial extent (default
        0.01). Must match what's passed to the Abaqus script.
    n_arrows : int, optional
        Approximate number of load arrows drawn (default 48). Cosmetic only.
    height : int, optional
        Figure height in px (default 600).

    RETURNS
    -------
    fig : plotly.graph_objects.Figure

    RAISES
    ------
    Same as select_end_faces() / viz.build_mesh_figure().

    EXAMPLE
    -------
    >>> verts, faces = load_STL("block.stl")
    >>> fig = build_load_preview_figure(faces, verts, axis="z", load=100.0)
    >>> st.plotly_chart(fig)
    """
    # ------ check input data ------
    faces = np.asarray(faces, dtype=int)
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(f"build_load_preview_figure(): faces must be (M, 3), got {faces.shape}.")
    verts = np.asarray(verts, dtype=float)
    if verts.ndim != 2 or verts.shape[1] != 3:
        raise ValueError(f"build_load_preview_figure(): verts must be (N, 3), got {verts.shape}.")
    # ----- select the top and bottom faces ------
    # BC_info_dict is a dict with keys "top", "bottom", "axis", "extent", that define the two extreme faces along the chosen axis. 
    # The top face is where the load is applied, the bottom face is where the roller and pins are.
    BC_info_dict= select_end_faces(verts, axis, tol_frac=tol_frac)
    ax = BC_info_dict["axis"]
    t1, t2 = [i for i in range(3) if i != ax]
    axis_label = "XYZ"[ax]

    # ---- base structure -------------------------------------------------
    fig = viz.build_mesh_figure(faces, verts, show_normal_colorscale=True)

    # ---- arrow geometry ------------------------------------------------
    # bbox is the bounding box of the STL (in 3 dimensions)
    bbox = verts.max(axis=0) - verts.min(axis=0)
    # define arrow length as a fraction of the largest dimension of the bounding box.
    arrow_len = 0.15 * bbox.max()
    if arrow_len <= 0:
        arrow_len = 1.0

    # find points where arrows will be drawn on the top face.
    top_pts = _sample_face_points(np.unique(verts[BC_info_dict["top"]], axis=0), t1, t2, n_arrows)

    # sense = +1 when the arrow points along +axis.
    sense = -1.0 if compression else 1.0

    # Define the tail and tip of the arrows. (from where the arrow starts to where it points to)
    tail, tip = top_pts.copy(), top_pts.copy()
    if compression:
        tail[:, ax] += arrow_len
    else:
        tip[:, ax] += arrow_len

    # draw the arrows as a set of lines, and a set of cones (for the arrowheads).
    # define the x, y, z coordinates of the lines that make up the arrows. The None values are used to separate the individual arrows in the plotly scatter3d trace.
    sx, sy, sz = [], [], []
    for p0, p1 in zip(tail, tip):
        sx += [p0[0], p1[0], None]
        sy += [p0[1], p1[1], None]
        sz += [p0[2], p1[2], None]

    #draw the lines as a scatter3d trace. 
    sign_txt = "compression" if compression else "tension"
    fig.add_trace(go.Scatter3d(x=sx, y=sy, z=sz, 
                                mode="lines",
                                line=dict(color="crimson", width=4),
                                hoverinfo="skip", 
                                name=f"load ({sign_txt})", 
                                showlegend=True,
    ))

    # draw the tips of the arrows as a cone trace. THANK YOU AI FOR THIS FUNCTION ><
    cone_vec = np.zeros_like(tip)
    cone_vec[:, ax] = sense * arrow_len
    u, v, w = cone_vec[:, 0], cone_vec[:, 1], cone_vec[:, 2]
    # calculate the load per node for the hover text. The total load is divided by the number of nodes on the top face, which is given by the sum of the boolean array BC_info_dict["top"]. If there are no nodes on the top face, we avoid division by zero by using max(int(BC_info_dict["top"].sum()), 1). This ensures that per_node is always a valid number.
    per_node = load / max(int(BC_info_dict["top"].sum()), 1)
    fig.add_trace(go.Cone(x=tip[:, 0], y=tip[:, 1], z=tip[:, 2], 
                            u=u, v=v, w=w,
                            sizemode="raw", 
                            sizeref=0.35,# * arrow_len, 
                            anchor="tip",
                            showscale=False, 
                            colorscale=[[0, "crimson"], [1, "crimson"]],
                            hovertemplate=(f"Set-Top - total {load:g} along {'-' if compression else '+'}"
                                        f"{axis_label}<br>~{per_node:.3g} per node<extra></extra>"),
                            name="load", 
                            showlegend=False,
    ))

    # ---- supports on the bottom face -----------------------------------
    # find points where the roller BC will be drawn on the bottom face.
    bot_pts = _sample_face_points(np.unique(verts[BC_info_dict["bottom"]], axis=0), t1, t2, n_arrows)

    fig.add_trace(go.Scatter3d(x=bot_pts[:, 0], y=bot_pts[:, 1], z=bot_pts[:, 2], 
                                mode="markers",
                                marker=dict(size=3.5, color="royalblue", symbol="x"),
                                name=f"roller: U{ax + 1} = 0",
                                hovertemplate=f"Set-Bottom - U{ax + 1} = 0<extra></extra>",
    ))

    # ---- pins on the bottom face -----------------------------------
    idx_a, idx_b = pick_rbm_nodes(verts, BC_info_dict["bottom"], ax)
    rbm = verts[[idx_a, idx_b]]
    fig.add_trace(go.Scatter3d(
        x=rbm[:, 0], y=rbm[:, 1], z=rbm[:, 2], mode="markers",
        marker=dict(size=8, color="darkorange", symbol="diamond",
                    line=dict(color="black", width=1)),
        name="rigid-body pins",
        text=[f"pin A - U{t1 + 1} = U{t2 + 1} = 0", f"pin B - U{t2 + 1} = 0"],
        hovertemplate="%{text}<extra></extra>",
    ))

    # ------ finalize the figure layout -----------------------------------
    fig.update_layout(
        height=height,
        title=(f"Load case: {load:g} in {sign_txt} along {axis_label}"
               f"  |  face slab = {100 * tol_frac:g} % of height"),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0),
    )
    return fig, rbm


# =====================================================================
# 6) _axis_index
# =====================================================================
_AXIS_NAMES = {"x": 0, "y": 1, "z": 2, "0": 0, "1": 1, "2": 2}

def _axis_index(axis) -> int:
    """
    ============================================================================
    6) _AXIS_INDEX
    small utility to normalise an axis given as "x"/"y"/"z" (any case) or 0/1/2 into an
    integer index.
    ============================================================================

    PARAMETERS
    ----------
    axis : str or int
        "x", "y", "z" (case-insensitive) or 0, 1, 2.

    RETURNS
    -------
    int
        0, 1 or 2.
    """
    # if the user passed an integer, just check it's in  the right range and return it.
    if isinstance(axis, (int, np.integer)):
        if int(axis) in (0, 1, 2):
            return int(axis)
        logger.debug(f"axis_index(): axis must be 0, 1 or 2, got {axis}.")
        raise ValueError(f"axis_index(): axis must be 0, 1 or 2, got {axis}.")
    # if the user passed a string, normalise it and look it up in the dict.
    key = str(axis).strip().lower()
    if key not in _AXIS_NAMES:
        logger.debug(f"axis_index(): unknown axis {axis!r}. Use 'x', 'y', 'z' or 0/1/2.")
        raise ValueError(f"axis_index(): unknown axis {axis!r}. Use 'x', 'y', 'z' or 0/1/2.")
    return _AXIS_NAMES[key]


# =====================================================================
# 7) select_end_faces
# =====================================================================
def select_end_faces(verts: np.ndarray,
                    axis: str,
                    tol_frac: float = 0.01) -> dict:
    """
    ============================================================================
    7) SELECT_END_FACES
    Splits a point cloud into the "bottom" and "top" slabs along one axis:
    every point within `tol_frac` x (axial extent) of the bounding-box
    minimum / maximum along that axis.

    This mirrors the rule generate_static_sim.py uses to build
    Set-Bottom / Set-Top inside Abaqus (there via
    MeshNodeArray.getByBoundingBox, here via a mask - same band). It is a
    mirror, not the real thing: Abaqus runs its own copy on the tet mesh.
    ============================================================================

    PARAMETERS
    ----------
    verts : (N, 3) ndarray
        Point coordinates (STL vertices, or mesh nodes).
    axis : str or int
        Load axis - see _axis_index().
    tol_frac : float, optional
        Half-thickness of each face slab, as a fraction of the total
        extent along `axis`. Default 0.01 (= 1 % of the height at each
        end). Must be > 0.

    RETURNS
    -------
    dict with keys
        "axis"        : int, resolved axis index
        "bottom"      : (N,) bool mask
        "top"         : (N,) bool mask
        "lo", "hi"    : float, bounding-box extremes along `axis`
        "extent"      : float, hi - lo
        "tol"         : float, absolute slab thickness used

    RAISES
    ------
    ValueError
        If `verts` isn't (N, 3), if tol_frac <= 0, or if the structure is
        degenerate (zero extent) along the requested axis.
    """
    # ------ check inputs ------
    verts = np.asarray(verts, dtype=float)
    if verts.ndim != 2 or verts.shape[1] != 3:
        raise ValueError(f"select_end_faces(): verts must be (N, 3), got {verts.shape}.")
    if tol_frac <= 0:
        raise ValueError(f"select_end_faces(): tol_frac must be > 0, got {tol_frac}.")

    # ----- find the two end slabs ------
    # grab the coordintes on the axis asked
    ax = _axis_index(axis)
    coord = verts[:, ax]
    # find min/max along that axis
    lo, hi = float(coord.min()), float(coord.max())
    extent = hi - lo
    # a flat structure has no two opposite faces to grip. Without this guard
    # tol comes out 0, and both `coord <= lo` and `coord >= hi` are then True
    # for every point - so bottom and top would silently come back as the
    # whole structure, held and loaded at once.
    if extent <= 0:
        raise ValueError(
            f"select_end_faces(): the geometry has zero extent along axis {ax} - "
            "there are no two opposite faces to load."
        )

    # calculate absolute tolerance from the fraction and the extent
    tol = tol_frac * extent

    # find the points that fall within the tolerance band at each end
    bottom = coord <= lo + tol  # creates a boolean mask of the points that are within the tolerance !
    top = coord >= hi - tol

    #return the info as a dict, so the caller can use the masks and the bounding-box info.
    logger.debug(
        f"select_end_faces(): axis={ax}, extent={extent:.6g}, tol={tol:.6g}, "
        f"{int(bottom.sum())} bottom pts, {int(top.sum())} top pts"
    )
    return {"axis": ax, "bottom": bottom, "top": top,
            "lo": lo, "hi": hi, "extent": extent, "tol": tol}


# =====================================================================
# 8) pick_rbm_nodes
# =====================================================================
def pick_rbm_nodes(verts: np.ndarray,
                    bottom_mask: np.ndarray,
                    axis: str,
                    min_dist_frac: float = 0.2) -> tuple:
    """
    ============================================================================
    8) PICK_RBM_NODES
    Picks the two bottom-face points used to remove the rigid-body modes
    that a roller support leaves free.

    The first point (A) is the one closest to the middle of the bottom slab.
    The second point (B) is the one that sits far enough away from A along the
    first transverse axis (t1) to give a lever arm, and that strays least
    from the t1 axis through A.
    ============================================================================

    PARAMETERS
    ----------
    verts : (N, 3) ndarray
        Point coordinates.
    bottom_mask : (N,) bool ndarray
        Which points belong to the bottom face (from select_end_faces()).
    axis : str or int
        Load axis - see _axis_index().
    min_dist_frac : float, optional
        The lever-arm floor: how far B must sit from A *along t1*, as a
        fraction of the face's t1 extent (default 0.2). This is what makes
        the constraint bite - alignment alone would happily return A's
        nearest neighbour, whose offset is nearly zero.

        Note A is central, so |d1| can't exceed about half the t1 extent;
        values much above 0.4 will fail on most faces.

    RETURNS
    -------
    (idx_a, idx_b) : tuple of int
        Indices *into verts* of the two chosen points. The DOF to fix at B
        is always t2.

    RAISES
    ------
    ValueError
        If the bottom mask selects fewer than 2 points, or if no point sits
        at least min_dist_frac of the face's t1 extent away from A along t1.
    """
    if min_dist_frac <= 0:
        logger.error(f"pick_rbm_nodes(): min_dist_frac must be > 0, got {min_dist_frac}.")
        raise ValueError(f"pick_rbm_nodes(): min_dist_frac must be > 0, got {min_dist_frac}.")
    elif min_dist_frac >= 0.45:
        logger.error(f"pick_rbm_nodes(): min_dist_frac must be < 0.45, got {min_dist_frac}.")
        raise ValueError(f"pick_rbm_nodes(): min_dist_frac must be < 0.45, got {min_dist_frac}.")


    verts = np.asarray(verts, dtype=float)
    ax = _axis_index(axis)
    # define the two transverse axes : i.e. the two that aren't the load axis.
    t1, t2 = [i for i in range(3) if i != ax]

    # get the indices of the points that are in the bottom slab
    idx = np.flatnonzero(bottom_mask)   #returns the indices of the elements that are non-zero in the flattened version of the input array
    if idx.size < 2:
        raise ValueError(
            "pick_rbm_nodes(): the bottom face slab holds fewer than 2 points - "
            "increase the face tolerance."
        )

    # grab the coordinates of those points (in the bottom slab)
    pts = verts[idx]

    # ---- pin A : the point closest to the middle of the bottom slab ----
    middle_point = np.mean(pts, axis=0)
    dist_to_middle = np.linalg.norm(pts - middle_point, axis=1)
    i_a = int(np.argmin(dist_to_middle))   # position within pts
    idx_a = int(idx[i_a])                  # position within verts

    # ---- pin B : offset from A along t1, hugging the t1 axis ----
    # in-plane offsets of every candidate from A. Signed; only magnitudes
    # matter below.
    distance_to_center = pts[:, t1] - pts[i_a, t1]      # how far B is from A along t1
    stray_from_axis = pts[:, t2] - pts[i_a, t2]         # how far B strays off the t1 axis

    # make a mask of the candidates that are far enough away from A along
    span_t1 = float(np.ptp(pts[:, t1]))   # np.ptp = "peak to peak", i.e. max - min
    min_dist = min_dist_frac * span_t1
    far_enough = np.abs(distance_to_center) >= min_dist

    if not far_enough.any():
        logger.error(
            f"pick_rbm_nodes(): no bottom-face point sits {min_dist:.4g} from the "
            f"central pin along {'XYZ'[t1]}. Lower min_dist_frac or widen the face tolerance."
        )
        raise ValueError(
                f"pick_rbm_nodes(): no bottom-face point sits {min_dist:.4g} from the "
                f"central pin along {'XYZ'[t1]}. Lower min_dist_frac or widen the face tolerance."
        )

    # find the candidate and that strays least from the t1 axis through A.
    i_b = int(np.argmin(np.abs(stray_from_axis[far_enough])))
    idx_b = int(idx[far_enough][i_b])

    return idx_a, idx_b

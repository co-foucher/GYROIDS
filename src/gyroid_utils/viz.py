import numpy as np
import plotly.graph_objects as go
from .logger import logger
from plotly.colors import sample_colorscale


"""
#=====================================================================================================================
0 - (reserved)
1 - save_mesh_as_html
2 - plot_histogram
3 - twod_view_of_matrix
4 - view_mesh
#=====================================================================================================================
"""

# =====================================================================
# 1) _build_mesh_figure / build_mesh_figure / save_mesh_as_html
# =====================================================================
def _build_mesh_figure(faces: np.ndarray,
                       verts: np.ndarray,
                       show_normal_colorscale: bool = False,
                       show_flat_colorscale: bool = False,
                       show_random_colorscale: bool = False,
                       show_curvature_colorscale: bool = False):
    """
    Shared core of build_mesh_figure()/save_mesh_as_html(): validates
    faces/verts, resolves the show_*_colorscale flag precedence, reduces
    faces if needed, computes facecolor, and returns the Mesh3d go.Figure.
    No file I/O and no fig.show()/fig.write_html() here - callers decide
    what to do with the returned figure. See save_mesh_as_html()'s
    docstring for parameter/exception details; this raises the same
    TypeError/ValueError/RuntimeError, at the same points.
    """
    # ----------------------------------------------
    # Validate input
    # ----------------------------------------------
    if faces is None or verts is None:
        logger.error("build_mesh_figure(): faces or verts is None.")
        raise TypeError("build_mesh_figure(): faces and verts must not be None.")

    if len(faces) == 0:
        logger.error("build_mesh_figure(): No faces provided.")
        raise ValueError("build_mesh_figure(): faces is empty, nothing to visualize.")

    if show_curvature_colorscale ==False and show_flat_colorscale == False and show_random_colorscale == False and show_normal_colorscale == False:
        logger.warning("No colorscale option selected. Defaulting to normal colorscale.")
        show_normal_colorscale = True
    elif sum([show_curvature_colorscale, show_flat_colorscale, show_random_colorscale, show_normal_colorscale]) > 1:
        logger.warning("Multiple colorscale options selected. Defaulting to normal colorscale.")
        show_normal_colorscale = True
        show_flat_colorscale = False
        show_random_colorscale = False
        show_curvature_colorscale = False

    logger.debug(f"Input mesh: {verts.shape[0]} vertices, {faces.shape[0]} faces")

    # =========================================================
    # 0) FACE DECIMATION
    # =========================================================
    target_faces = 5_000_000

    if faces.shape[0] > target_faces:
        logger.info(
            f"Reducing faces: {faces.shape[0]} → {target_faces} (centroid-based filtering)"
        )

        try:
            centroids = verts[faces].mean(axis=1)
            centroid_norm = np.linalg.norm(centroids, axis=1)

            keep_idx = np.argpartition(centroid_norm, target_faces - 1)[:target_faces]
            keep_idx = keep_idx[np.argsort(centroid_norm[keep_idx])]
            faces = faces[keep_idx]

            logger.debug(f"Faces reduced to: {faces.shape[0]}")
        except Exception as e:
            logger.error(f"Face decimation failed: {e}", exc_info=True)


    # =========================================================
    # 4) BUILD PLOTLY FIGURE
    # =========================================================
    logger.debug("Building Plotly figure...")

    try:
        # Per-face random colors for better visual differentiation
        if show_normal_colorscale:
            face_normals = np.cross(
                verts[faces[:, 1]] - verts[faces[:, 0]],
                verts[faces[:, 2]] - verts[faces[:, 0]]
            )
            norm_magnitudes = np.linalg.norm(face_normals, axis=1, keepdims=True)
            norm_magnitudes[norm_magnitudes == 0] = 1.0
            face_normals /= norm_magnitudes

            facecolor = [
                f"rgb({int((n[0]+1)/2*255)},{int((n[1]+1)/2*255)},{int((n[2]+1)/2*255)})"
                for n in face_normals
            ]
        elif show_random_colorscale:
            try:
                rng = np.random.default_rng()
                cols = rng.integers(0, 255, size=(faces.shape[0], 3), dtype=np.uint8)
                facecolor = [f"rgb({r},{g},{b})" for r, g, b in cols]
            except Exception as e:
                logger.error(f"Random colorscale generation failed: {e}", exc_info=True)
                raise RuntimeError("build_mesh_figure(): failed to generate random colorscale") from e

        elif show_flat_colorscale:
            # Mesh3d's facecolor needs one entry per face (a list/array),
            # not a single string - a bare string used to raise
            # "Invalid value of type 'builtins.str' ... facecolor" below.
            facecolor = ['lightblue'] * faces.shape[0]

        elif show_curvature_colorscale:
            # ---------------------------------------------------------
            # Vertex curvature from topological neighbors (mesh propagation)
            # Uses BFS to find k-nearest neighbors along mesh edges
            # avoids cross-wall artifacts on thin geometries
            # (surface variation from local covariance eigenvalues)
            # Then face curvature = mean of its 3 vertex curvatures.
            # ---------------------------------------------------------

            # Auto radius from mesh size if not provided
            bbox = verts.max(axis=0) - verts.min(axis=0)
            curvature_min_neighbors = 15  # minimum neighbors for curvature calc

            n_verts = verts.shape[0]
            vertex_curv = np.zeros(n_verts, dtype=np.float64)

            # Build vertex adjacency list from faces
            adj = [set() for _ in range(n_verts)]
            for f in faces:
                adj[f[0]].update([f[1], f[2]])
                adj[f[1]].update([f[0], f[2]])
                adj[f[2]].update([f[0], f[1]])

            # BFS to find k topological neighbors for each vertex
            def _find_topological_neighbors(start_vi, k):
                """
                ========================================================================
                _FIND_TOPOLOGICAL_NEIGHBORS
                Returns k nearest neighbors along mesh topology via BFS.
                ========================================================================

                PARAMETERS
                ----------
                start_vi : int
                    Index of the starting vertex.
                k : int
                    Number of neighbors to find.

                RETURNS
                -------
                neighbors : list[int]
                    Up to k vertex indices found via breadth-first search.
                """
                visited = {start_vi}
                queue = [start_vi]
                neighbors = []

                while queue and len(neighbors) < k:
                    vi = queue.pop(0) #create a queue for breadth first search

                    # Add unvisited neighbors to queue
                    for next_vi in adj[vi]:
                        if next_vi not in visited:
                            visited.add(next_vi)
                            neighbors.append(next_vi)
                            queue.append(next_vi)
                            if len(neighbors) >= k:
                                break
                return neighbors[:k]

            # Calculate curvature for each vertex based on topological neighbors
            for vi in range(n_verts):
                nbrs = _find_topological_neighbors(vi, k=curvature_min_neighbors)

                if len(nbrs) < max(3, curvature_min_neighbors):
                    # If too few neighbors, skip curvature calculation (will be zero)
                    continue

                # Calculate relative positions of neighbors
                P = verts[nbrs] - verts[vi]

                # Calculate covariance-like matrix
                C = (P.T @ P) / max(P.shape[0], 1)
                # Calculate eigenvalues of covariance matrix
                evals = np.linalg.eigvalsh(C)
                # Surface variation = smallest eigenvalue / sum of eigenvalues
                denom = float(evals.sum())
                if denom > 0:
                    # small when planar, larger in curved/rough regions
                    vertex_curv[vi] = float(evals[0] / denom)
                # Print progress every 10k verts
                if vi % 10000 == 0:
                    logger.info(f"Curvature: processed vertex {vi}/{n_verts}")

            # Clamp negative curvature values due to numerical issues
            vnorm = np.clip(vertex_curv, 0.0, None)
            # Normalize to [0, 1] using 95th percentile (less sensitive to outliers)
            p95 = np.percentile(vnorm, 95)
            if p95 > 0:
                vnorm = vnorm / p95
            vnorm = np.clip(vnorm, 0.0, 1.0)

            face_curv = vnorm[faces].mean(axis=1)  # average node curvature per face
            facecolor = sample_colorscale("Turbo", face_curv.tolist())

        mesh = go.Mesh3d(
            x=verts[:, 0],
            y=verts[:, 1],
            z=verts[:, 2],
            i=faces[:, 0],
            j=faces[:, 1],
            k=faces[:, 2],
            facecolor=facecolor if facecolor is not None else 'lightblue',
            opacity=1
        )

        fig = go.Figure(data=[mesh])

        fig.update_layout(
            scene=dict(
                xaxis=dict(visible=True),
                yaxis=dict(visible=True),
                zaxis=dict(visible=True),
                aspectmode='data',
            ),
            title="Mesh Preview (Reduced)"
        )
    except Exception as e:
        logger.error(f"Failed to build Plotly figure: {e}", exc_info=True)
        raise RuntimeError("build_mesh_figure(): failed to build Plotly figure") from e

    return fig


def build_mesh_figure(faces: np.ndarray,
                      verts: np.ndarray,
                      show_normal_colorscale: bool = False,
                      show_flat_colorscale: bool = False,
                      show_random_colorscale: bool = False,
                      show_curvature_colorscale: bool = False):
    """
    ============================================================================
    BUILD_MESH_FIGURE
    Builds the same Mesh3d go.Figure as save_mesh_as_html(), without ever
    touching disk. For callers that embed the figure directly (e.g.
    Streamlit's st.plotly_chart) - going through save_mesh_as_html() +
    reading the .html back would mean writing/re-parsing a multi-MB file
    (plotly.js embedded inline plus the mesh data) on every call just to
    get bytes you're about to hand to a renderer that already has its own
    Plotly runtime. Use save_mesh_as_html() when you actually want a
    standalone .html artifact on disk (e.g. the Library page's preview
    files); use this when you just want the figure object.
    ============================================================================

    PARAMETERS / RAISES
    --------------------
    Same as save_mesh_as_html() (see its docstring) minus `file_name` and
    `save`, which don't apply here.

    RETURNS
    -------
    fig : plotly.graph_objects.Figure

    EXAMPLE
    -------
    >>> fig = build_mesh_figure(faces, verts, show_normal_colorscale=True)
    >>> st.plotly_chart(fig)
    """
    return _build_mesh_figure(faces, verts, show_normal_colorscale, show_flat_colorscale,
                              show_random_colorscale, show_curvature_colorscale)


def save_mesh_as_html(faces: np.ndarray,
                      verts: np.ndarray,
                      file_name: str,
                      show_normal_colorscale: bool = False,
                      show_flat_colorscale: bool = False,
                      show_random_colorscale: bool = False,
                      show_curvature_colorscale: bool = False,
                      save: bool = True):
    """
    ============================================================================
    1) SAVE_MESH_AS_HTML
    Converts a mesh into a lightweight Plotly 3D HTML visualization.
    Handles face/edge reduction for performance.
    ============================================================================

    PARAMETERS
    ----------
    faces : (M, 3) ndarray
        Triangle indices.
    verts : (N, 3) ndarray
        Vertex coordinates.
    file_name : str
        Output HTML file name (without extension).
    show_normal_colorscale : bool
        If True, colors faces based on normal vectors.
    show_flat_colorscale : bool
        If True, colors faces with a flat color.
    show_random_colorscale : bool
        If True, colors faces with random colors.
    show_curvature_colorscale : bool
        If True, colors faces based on curvature (not implemented).
    save : bool
        If True, saves the HTML file. If False, displays the figure without saving.

    RETURNS
    -------
    None

    RAISES
    ------
    TypeError
        If faces or verts is None.
    ValueError
        If faces is empty.
    RuntimeError
        If the Plotly figure fails to build, the random colorscale fails to
        generate, or the HTML file fails to write.

    NOTES
    -----
    - Only one colorscale option should be True. If multiple or none are True,
      normal colorscale takes precedence. This is legacy behavior kept for
      backward compatibility with existing call sites.
    - The exported file loads plotly.js from a CDN (`include_plotlyjs="cdn"`)
      instead of embedding the ~4 MB library inline, so this both writes and
      loads noticeably faster - the tradeoff is that opening the .html needs
      internet access to render. Callers that just want the figure in memory
      (no disk round-trip at all) should use build_mesh_figure() instead.

    OUTPUT
    ------
    Creates a file:
        <file_name>.html

    EXAMPLE
    -------
    >>> save_mesh_as_html(faces, verts, "mesh_preview")
    """
    logger.info(f"Saving mesh visualization → '{file_name}.html'")

    fig = _build_mesh_figure(faces, verts, show_normal_colorscale, show_flat_colorscale,
                             show_random_colorscale, show_curvature_colorscale)

    # =========================================================
    # SAVE HTML FILE
    # =========================================================
    if save:
        try:
            out_path = f"{file_name}.html"
            fig.write_html(out_path, auto_open=False, include_plotlyjs="cdn")
            logger.info(f"HTML visualization saved → {out_path}")

        except Exception as e:
            logger.error(f"Failed to save HTML visualization: {e}", exc_info=True)
            raise RuntimeError(f"save_mesh_as_html(): failed to write '{out_path}'") from e
    else:
        fig.show()
        logger.info("HTML visualization displayed (not saved).")


#=====================================================================
#2) plot_histogram
#=====================================================================
def plot_histogram(face_areas, BINS=1000):
    """
    ============================================================================
    2) PLOT_HISTOGRAM
    Plots a PDF-like histogram of triangle areas using Plotly.
    ============================================================================

    PARAMETERS
    ----------
    face_areas : array-like
        List/array of triangle areas.
    BINS : int, optional
        Number of histogram bins (default = 1000).

    RETURNS
    -------
    None

    RAISES
    ------
    ValueError
        If face_areas is None or empty.
    RuntimeError
        If the histogram fails to compute or display.

    NOTES
    -----
    - Displays a line-plot representation of the PDF.

    EXAMPLE
    -------
    >>> plot_histogram(areas)
    """
    if face_areas is None or len(face_areas) == 0:
        logger.error("plot_histogram(): empty area array — nothing to plot.")
        raise ValueError("plot_histogram(): face_areas is empty.")

    logger.info(f"Plotting histogram for {len(face_areas)} triangle areas")

    try:
        hist = np.histogram(face_areas, bins=BINS)
        counts, bins = hist
        bin_centers = 0.5 * (bins[:-1] + bins[1:])
    except Exception as e:
        logger.error(f"Failed to compute histogram: {e}", exc_info=True)
        raise RuntimeError("plot_histogram(): failed to compute histogram") from e

    logger.debug(
        f"Histogram stats — min area: {face_areas.min()}, max area: {face_areas.max()}"
    )

    try:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=bin_centers,
            y=counts,
            mode='lines',
            name='PDF'
        ))

        fig.update_layout(
            title="Triangle Area Size Distribution (PDF)",
            xaxis_title="Triangle area",
            yaxis_title="Count"
        )

        fig.show()
        logger.info("Histogram displayed successfully.")
    except Exception as e:
        logger.error(f"Failed to display histogram: {e}", exc_info=True)


#=====================================================================
#3) twod_view_of_matrix
#=====================================================================

def twod_view_of_matrix(v: np.ndarray,
                        x: np.ndarray = None,
                        y: np.ndarray = None,
                        z: np.ndarray = None,
                        zmin=None,
                        zmax=None,
                        show: bool = True,
                        max_pixels: int = 30_000_000):
    """
    ============================================================================
    3) TWOD_VIEW_OF_MATRIX
    Creates a scrollable 2D heatmap visualization of a 3D scalar field v(x,y,z).
    ============================================================================

    PARAMETERS
    ----------
    v : (Nx, Ny, Nz) ndarray
        Scalar field (e.g., gyroid field values).
    x : (Nx, 1, 1) ndarray, optional
        X-coordinate grid. If None, uses np.arange(Nx).
    y : (1, Ny, 1) ndarray, optional
        Y-coordinate grid. If None, uses np.arange(Ny).
    z : (1, 1, Nz) ndarray, optional
        Z-coordinate grid. If None, uses np.arange(Nz).
    zmin, zmax : float or None, optional
        Color limits for the heatmap. If None, uses min/max from v.
    show : bool, optional
        If True (default), calls fig.show() as before. If False, returns
        the Plotly Figure instead of showing it - for callers that want to
        embed it themselves (e.g. Streamlit's st.plotly_chart), matching
        the save/show split already used by save_mesh_as_html().
    max_pixels : int, optional
        Pixel budget for the whole animation (default = 30,000,000). One
        Plotly "frame" is pre-built per displayed Z slice, and each frame
        costs Nx*Ny pixels - past a few hundred slices that payload is
        what freezes/crashes the notebook or browser tab. If
        Nx*Ny*Nz > max_pixels, every Z slice is no longer shown: instead
        only every `step`-th slice gets a frame, where `step` is the
        smallest value that brings the *frame count* (not the per-slice
        resolution) back under budget. X/Y resolution is never reduced -
        each shown slice is still full Nx*Ny detail, only the number of
        Z positions you can scrub to goes down. Pass a larger value (or
        float('inf')) to disable this and always build one frame per slice.

    RETURNS
    -------
    fig : plotly.graph_objects.Figure or None
        The figure, if show=False. Otherwise None (the figure is shown
        directly instead).

    RAISES
    ------
    ValueError
        If v is not 3D, or if the (x, y, z) grid shapes do not match v.

    NOTES
    -----
    - Builds one animation frame per (possibly strided) Z slice up front,
      so this can still get slow / produce a large payload for very fine
      X/Y grids even after striding - true whether shown directly or
      embedded via show=False. Lower max_pixels for a lighter animation.

    EXAMPLE
    -------
    >>> twod_view_of_matrix(v, x, y, z)
    >>> fig = twod_view_of_matrix(v, x, y, z, show=False)
    >>> twod_view_of_matrix(big_v, x, y, z, max_pixels=2_000_000)  # keep more Z slices
    """

    logger.info("Starting 2D visualization of 3D matrix.")

    # ------------------------------------------------------------------
    # Validate inputs
    # ------------------------------------------------------------------
    if v.ndim != 3:
        logger.error("twod_view_of_matrix(): v must be 3D (Nx, Ny, Nz).")
        raise ValueError(f"twod_view_of_matrix(): v must be 3D, got ndim={v.ndim}.")

    Nx, Ny, Nz = v.shape

    if x is None:
        x = np.arange(Nx)
    if y is None:
        y = np.arange(Ny)
    if z is None:
        z = np.arange(Nz)
    if x.ndim == 1 and y.ndim == 1 and z.ndim == 1:
        x, y, z = np.meshgrid(x, y, z, indexing='ij')

    if x.shape[0] != Nx or y.shape[1] != Ny or z.shape[2] != Nz:
        logger.error("Grid dimensions of (x,y,z) do not match v.shape.")
        raise ValueError("twod_view_of_matrix(): grid dimensions of (x,y,z) do not match v.shape.")

    logger.debug(f"Field resolution: {Nx} × {Ny} × {Nz}")

    # ------------------------------------------------------------------
    # Setup axes
    # ------------------------------------------------------------------
    x_axis = x[:, 0, 0]
    y_axis = y[0, :, 0]
    z_axis = z[0, 0, :]

    if zmin is None:
        zmin = float(np.min(v))
    if zmax is None:
        zmax = float(np.max(v))

    logger.debug(f"Color range: zmin={zmin}, zmax={zmax}")

    # ------------------------------------------------------------------
    # Decide which Z slices actually get a frame, based on max_pixels.
    # X/Y resolution is never touched - only how many Z positions are
    # shown gets reduced, by taking every `step`-th slice instead of all
    # Nz of them.
    # ------------------------------------------------------------------
    total_pixels = Nx * Ny * Nz
    per_slice_pixels = Nx * Ny

    if total_pixels > max_pixels:
        max_frames = max(1, int(max_pixels // per_slice_pixels))
        step = -(-Nz // max_frames)  # ceil(Nz / max_frames) via integer arithmetic
        slice_indices = list(range(0, Nz, step))
        logger.warning(
            f"Volume has {total_pixels:,} pixels ({Nx}×{Ny}×{Nz}), over "
            f"max_pixels={max_pixels:,}. Showing every {step} of {Nz} Z "
        )
    else:
        step = 1
        slice_indices = list(range(Nz))

    # Start at first (shown) slice
    k0 = slice_indices[0]

    # ------------------------------------------------------------------
    # Build frames for animation
    # ------------------------------------------------------------------
    frames = [
        go.Frame(
            data=[go.Heatmap(
                x=x_axis,
                y=y_axis,
                z=v[:, :, k].T,
                colorscale="Portland",
                zmin=zmin,
                zmax=zmax
            )],
            name=str(k)
        )
        for k in slice_indices
    ]

    logger.info(f"Generated {len(frames)} animation frame(s) (of {Nz} Z slices).")

    # ------------------------------------------------------------------
    # Build figure
    # ------------------------------------------------------------------
    fig = go.Figure(
        data=[go.Heatmap(
            x=x_axis,
            y=y_axis,
            z=v[:, :, k0].T,
            colorscale="Portland",
            zmin=zmin,
            zmax=zmax
        )],
        layout=go.Layout(
            title=f"Gyroid field heatmap (z = {z_axis[k0]:.3f})",
            xaxis_title="X",
            yaxis_title="Y",
            yaxis=dict(scaleanchor="x", scaleratio=1),
            width=800,
            height=650,
            updatemenus=[
                {
                    "type": "buttons",
                    "direction": "left",
                    "x": 0.0, "y": 1.15,
                    "buttons": [
                        {"label": "Play", "method": "animate",
                         "args": [None, {
                             "fromcurrent": True,
                             "frame": {"duration": 60, "redraw": True}
                         }]},
                        {"label": "Pause", "method": "animate",
                         "args": [[None], {
                             "mode": "immediate",
                             "frame": {"duration": 0, "redraw": False}
                         }]}
                    ]
                }
            ]
        ),
        frames=frames
    )

    # ------------------------------------------------------------------
    # Slider for z-slices
    # ------------------------------------------------------------------
    fig.update_layout(
        sliders=[{
            "active": 0,
            "pad": {"t": 60},
            "currentvalue": {"prefix": "z = "},
            "steps": [
                {
                    "label": f"{z_axis[k]:.3f}",
                    "method": "animate",
                    "args": [
                        [str(k)],
                        {"mode": "immediate",
                         "frame": {"duration": 0, "redraw": True}}
                    ],
                }
                for k in slice_indices
            ]
        }]
    )

    if not show:
        logger.info("Returning heatmap figure (show=False).")
        return fig

    logger.info("Displaying interactive heatmap viewer.")
    fig.show()


# =====================================================================
# 4) view_mesh
# =====================================================================
def view_mesh(faces, verts, show_normal_colorscale: bool = True,):
    """
    ============================================================================
    4) VIEW_MESH
    Converts a mesh into a lightweight Plotly 3D HTML visualization and
    displays it without saving to disk.
    ============================================================================

    PARAMETERS
    ----------
    faces : (M, 3) ndarray
        Triangle indices.
    verts : (N, 3) ndarray
        Vertex coordinates.
    show_normal_colorscale : bool, optional
        If True (default), colors faces based on normal vectors.

    RETURNS
    -------
    None

    EXAMPLE
    -------
    >>> view_mesh(faces, verts)
    """
    save_mesh_as_html(faces, verts, "nop", show_normal_colorscale=show_normal_colorscale, save = False)

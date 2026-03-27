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
# 1) save_mesh_as_html
# =====================================================================
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

    NOTE: Only one colorscale option should be True. If multiple or none are True, normal colorscale takes precedence.
    IT is a shitty system but legacy code is legacy code.

    OUTPUT
    ------
    Creates a file:
        <file_name>.html

    EXAMPLE
    -------
    >>> save_mesh_as_html(faces, verts, "mesh_preview")
    """
    logger.info(f"Saving mesh visualization → '{file_name}.html'")

    # ----------------------------------------------
    # Validate input
    # ----------------------------------------------
    if faces is None or verts is None:
        logger.error("save_mesh_as_html(): faces or verts is None.")
        return

    if len(faces) == 0:
        logger.warning("save_mesh_as_html(): No faces provided. Export aborted.")
        return

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
            except Exception:
                facecolor = None

        elif show_flat_colorscale:
            facecolor = 'lightblue'
        
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
                """Returns k nearest neighbors along mesh topology via BFS"""
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
        return

    # =========================================================
    # SAVE HTML FILE
    # =========================================================
    if save:
        try:
            out_path = f"{file_name}.html"
            fig.write_html(out_path, auto_open=False)
            logger.info(f"HTML visualization saved → {out_path}")
        except Exception as e:
            logger.error(f"Failed to save HTML visualization: {e}", exc_info=True)
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

    NOTES
    -----
    - Uses 1000 bins.
    - Displays a line-plot representation of the PDF.

    EXAMPLE
    -------
    >>> plot_histogram(areas)
    """
    if face_areas is None or len(face_areas) == 0:
        logger.warning("plot_histogram(): empty area array — nothing to plot.")
        return

    logger.info(f"Plotting histogram for {len(face_areas)} triangle areas")

    try:
        hist = np.histogram(face_areas, bins=BINS)
        counts, bins = hist
        bin_centers = 0.5 * (bins[:-1] + bins[1:])
    except Exception as e:
        logger.error(f"Failed to compute histogram: {e}", exc_info=True)
        return

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
#2) twod_view_of_matrix
#=====================================================================

def twod_view_of_matrix(v: np.ndarray,
                        x: np.ndarray = None,
                        y: np.ndarray = None,
                        z: np.ndarray = None,
                        zmin=None,
                        zmax=None):
    """
    ============================================================================
    2) TWOD_VIEW_OF_MATRIX
    Creates a scrollable 2D heatmap visualization of a 3D scalar field v(x,y,z).
    ============================================================================
    
    PARAMETERS
    ----------
    v : (Nx, Ny, Nz) ndarray
        Scalar field (e.g., gyroid field values).
    x : (Nx, 1, 1) ndarray
        X-coordinate grid.
    y : (1, Ny, 1) ndarray
        Y-coordinate grid.
    z : (1, 1, Nz) ndarray
        Z-coordinate grid.
    zmin, zmax : float or None
        Color limits for the heatmap. If None, uses min/max from v.
    
    RETURNS
    -------
    None (shows Plotly interactive viewer)
    """

    logger.info("Starting 2D visualization of 3D matrix.")

    # ------------------------------------------------------------------
    # Validate inputs
    # ------------------------------------------------------------------
    if v.ndim != 3:
        logger.error("twod_view_of_matrix(): v must be 3D (Nx, Ny, Nz).")
        return

    Nx, Ny, Nz = v.shape

    if x is None:
        x = np.arange(Nx)
    if y is None:
        y = np.arange(Ny)
    if z is None:
        z = np.arange(Nz)
    x, y, z = np.meshgrid(x, y, z, indexing='ij')

    if x.shape[0] != Nx or y.shape[1] != Ny or z.shape[2] != Nz:
        logger.error("Grid dimensions of (x,y,z) do not match v.shape.")
        return

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

    # Start at first slice
    k0 = 0

    # ------------------------------------------------------------------
    # Build frames for animation
    # ------------------------------------------------------------------
    frames = [
        go.Frame(
            data=[go.Heatmap(
                x=x_axis,
                y=y_axis,
                z=v[:, :, k].T,
                colorscale="Viridis",
                zmin=zmin,
                zmax=zmax
            )],
            name=str(k)
        )
        for k in range(Nz)
    ]

    logger.info("Generated all animation frames.")

    # ------------------------------------------------------------------
    # Build figure
    # ------------------------------------------------------------------
    fig = go.Figure(
        data=[go.Heatmap(
            x=x_axis,
            y=y_axis,
            z=v[:, :, k0].T,
            colorscale="Viridis",
            zmin=zmin,
            zmax=zmax
        )],
        layout=go.Layout(
            title=f"Gyroid field heatmap (z = {z_axis[k0]:.3f})",
            xaxis_title="X",
            yaxis_title="Y",
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
            "active": k0,
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
                for k in range(Nz)
            ]
        }]
    )

    logger.info("Displaying interactive heatmap viewer.")
    fig.show()



# =====================================================================
# 4) view_mesh
# =====================================================================
def view_mesh(faces, verts, show_normal_colorscale: bool = True,):
    """
    ============================================================================
    4) view_mesh
    Converts a mesh into a lightweight Plotly 3D HTML visualization.
    Handles face/edge reduction for performance.
    ============================================================================
    """
    save_mesh_as_html(faces, verts, "nop", show_normal_colorscale=show_normal_colorscale, save = False)


























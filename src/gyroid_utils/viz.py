import numpy as np
import plotly.graph_objects as go
from .logger import logger
import plotly.graph_objects as go

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
def save_mesh_as_html(faces, verts, file_name, save: bool = True):
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
    # 1) BUILD EDGES
    # =========================================================
    try:
        e01 = faces[:, [0, 1]]
        e12 = faces[:, [1, 2]]
        e20 = faces[:, [2, 0]]
        edges = np.vstack([e01, e12, e20])
        edges.sort(axis=1)
        edges = np.unique(edges, axis=0)

        logger.debug(f"Extracted {edges.shape[0]} unique edges.")
    except Exception as e:
        logger.error(f"Failed to build edges: {e}", exc_info=True)
        return

    # If there are *too many* edges for Plotly/Edge to handle, export a
    # lightweight Three.js HTML viewer which scales much better for large
    # line counts. This avoids browser crashes when plotting millions of
    # edges with Plotly's higher-level renderer.
    try:
        export_threshold = 150_000
        if edges.shape[0] > export_threshold:
            logger.info(
                f"Large edge count ({edges.shape[0]}) — trying PyVista exporter for '{file_name}.html'"
            )

            # First try PyVista-based export; if it succeeds, return immediately.
            try:
                if _write_pyvista_html(file_name, verts, faces, edges, show_mesh=True):
                    return
            except Exception:
                logger.exception("PyVista export attempt failed — falling back to Plotly.")
    except Exception:
        logger.exception("Export selection failed — falling back to Plotly.")
    # =========================================================
    # 2) EDGE DECIMATION
    # =========================================================
    target_edges = 100_000

    if edges.shape[0] > target_edges:
        logger.info(
            f"Reducing edges: {edges.shape[0]} → {target_edges} (distance-based sampling)"
        )
        try:
            v_norm = np.linalg.norm(verts, axis=1)
            scores = np.minimum(v_norm[edges[:, 0]], v_norm[edges[:, 1]])

            keep_idx = np.argpartition(scores, target_edges - 1)[:target_edges]
            keep_idx = keep_idx[np.argsort(scores[keep_idx])]
            edges = edges[keep_idx]
        except Exception as e:
            logger.error(f"Edge decimation failed: {e}", exc_info=True)

    logger.debug(f"Final edge count for plot: {edges.shape[0]}")

    # =========================================================
    # 3) PREPARE EDGES FOR PLOTTING
    # =========================================================
    try:
        a, b = edges[:, 0], edges[:, 1]
        E = len(edges)

        edge_x = np.empty(3 * E)
        edge_y = np.empty(3 * E)
        edge_z = np.empty(3 * E)

        edge_x[0::3] = verts[a, 0]; edge_x[1::3] = verts[b, 0]; edge_x[2::3] = np.nan
        edge_y[0::3] = verts[a, 1]; edge_y[1::3] = verts[b, 1]; edge_y[2::3] = np.nan
        edge_z[0::3] = verts[a, 2]; edge_z[1::3] = verts[b, 2]; edge_z[2::3] = np.nan
    except Exception as e:
        logger.error(f"Preparing edges for plot failed: {e}", exc_info=True)
        return

    # =========================================================
    # 4) BUILD PLOTLY FIGURE
    # =========================================================
    logger.debug("Building Plotly figure...")

    try:
        mesh = go.Mesh3d(
            x=verts[:, 0],
            y=verts[:, 1],
            z=verts[:, 2],
            i=faces[:, 0],
            j=faces[:, 1],
            k=faces[:, 2],
            color='lightblue',
            opacity=1
        )

        edges_plot = go.Scatter3d(
            x=edge_x, y=edge_y, z=edge_z,
            mode='lines',
            line=dict(color='black', width=1),
            showlegend=False
        )

        fig = go.Figure(data=[mesh, edges_plot])

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
            out_path = f"{file_name}"
            fig.write_html(out_path, auto_open=True)
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
                        x: np.ndarray,
                        y: np.ndarray,
                        z: np.ndarray,
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
def view_mesh(faces, verts):
    """
    ============================================================================
    4) view_mesh
    Converts a mesh into a lightweight Plotly 3D HTML visualization.
    Handles face/edge reduction for performance.
    ============================================================================
    """
    save_mesh_as_html(faces, verts, "nop", save = False)



# ---------------------------------------------------------------------------
def _write_pyvista_html(file_name, verts, faces, edges, show_mesh=True):
    """
    Try to export an interactive HTML via PyVista/Panel. Returns True on success,
    False if PyVista isn't available or export failed.
    """
    try:
        import pyvista as pv # type: ignore
    except Exception:
        logger.debug("PyVista not available; skipping PyVista export.")
        return False

    try:
        verts = np.asarray(verts, dtype=float)
        faces = np.asarray(faces, dtype=np.int64)
        edges = np.asarray(edges, dtype=np.int64)

        # PyVista expects faces as [n, v0, v1, v2, n, v0, v1, v2, ...]
        faces_flat = np.hstack([np.full((faces.shape[0], 1), 3, dtype=np.int64), faces]).ravel()
        mesh = pv.PolyData(verts, faces_flat)

        # Build lines as a PolyData
        E = edges.shape[0]
        line_positions = np.empty((E * 2, 3), dtype=float)
        line_positions[0::2] = verts[edges[:, 0]]
        line_positions[1::2] = verts[edges[:, 1]]

        idx = np.arange(E * 2, dtype=np.int64)
        lines = np.empty(E * 3, dtype=np.int64)
        lines[0::3] = 2
        lines[1::3] = idx[0::2]
        lines[2::3] = idx[1::2]

        lines_mesh = pv.PolyData(line_positions, lines=lines)

        pl = pv.Plotter(off_screen=True)
        if show_mesh:
            pl.add_mesh(mesh, color='lightblue', opacity=1.0)
        pl.add_mesh(lines_mesh, color='black', line_width=1)
        pl.set_background('white')

        out_path = f"{file_name}.html"
        try:
            # Preferred: export interactive HTML (requires Panel/pyvista>=0.37)
            pl.export_html(out_path)
            logger.info(f"PyVista HTML visualization saved → {out_path}")
            return True
        except Exception:
            logger.exception("PyVista export_html failed; attempting screenshot fallback.")
            try:
                img_path = f"{file_name}.png"
                pl.show(screenshot=img_path, auto_close=True)
                html = f"<html><body style='margin:0'><img src='{img_path}' style='width:100%;height:auto;display:block'></body></html>"
                with open(out_path, 'w', encoding='utf-8') as f:
                    f.write(html)
                logger.info(f"PyVista screenshot HTML saved → {out_path}")
                return True
            except Exception:
                logger.exception("PyVista fallback export also failed.")
                return False
    except Exception:
        logger.exception("Failed to export with PyVista.")
        return False
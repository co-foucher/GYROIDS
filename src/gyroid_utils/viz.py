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
def save_mesh_as_html(faces, verts, file_name):
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
        threejs_threshold = 150_000
        if edges.shape[0] > threejs_threshold:
            logger.info(
                f"Large edge count ({edges.shape[0]}) — using Three.js exporter for '{file_name}.html'"
            )
            _write_threejs_html(file_name, verts, faces, edges)
            return
    except Exception:
        # Fall back to existing behaviour if Three.js path fails.
        logger.exception("Three.js export attempt failed — falling back to Plotly.")

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
    try:
        out_path = f"{file_name}.html"
        fig.write_html(out_path, auto_open=True)
        logger.info(f"HTML visualization saved → {out_path}")
    except Exception as e:
        logger.error(f"Failed to save HTML visualization: {e}", exc_info=True)


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
    logger.info(f"Viewing mesh")

    # ----------------------------------------------
    # Validate input
    # ----------------------------------------------
    if faces is None or verts is None:
        logger.error("view_mesh(): faces or verts is None.")
        return

    if len(faces) == 0:
        logger.warning("view_mesh(): No faces provided. Export aborted.")
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

    fig.show()


# ---------------------------------------------------------------------------
# Three.js HTML exporter (fast for many line segments)
# ---------------------------------------------------------------------------
def _write_threejs_html(file_name, verts, faces, edges, show_mesh=True):
        """
        Write a minimal standalone HTML file that renders the mesh and edges
        using Three.js. This scales much better for large numbers of line
        segments than Plotly's higher-level Scatter3d in some browsers.
        """
        try:
                verts = np.asarray(verts, dtype=float)
                faces = np.asarray(faces, dtype=np.int32)
                edges = np.asarray(edges, dtype=np.int32)

                # Prepare flat arrays for JS
                # Line positions: for each edge, two vertices (x,y,z)
                E = edges.shape[0]
                line_positions = np.empty((E * 2, 3), dtype=float)
                line_positions[0::2] = verts[edges[:, 0]]
                line_positions[1::2] = verts[edges[:, 1]]

                verts_flat = verts.reshape(-1).tolist()
                faces_flat = faces.reshape(-1).tolist()
                line_pos_flat = line_positions.reshape(-1).tolist()

                html = f"""<!doctype html>
<html>
<head>
    <meta charset="utf-8">
    <title>Mesh Preview</title>
    <style>body{{margin:0}}canvas{{width:100%;height:100%}}#info{{position:absolute;top:6px;left:6px;color:#fff;z-index:1}}</style>
</head>
<body>
<div id="info">Drag to rotate, scroll to zoom</div>
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r152/three.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/three@0.152.0/examples/js/controls/OrbitControls.js"></script>
<script>
const verts = new Float32Array({line_pos_size});
// placeholder
</script>
<script>
// Data arrays
const vertices = new Float32Array({verts_array});
const faceIdx = new Uint32Array({faces_array});
const linePositions = new Float32Array({lines_array});

// Scene
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(60, window.innerWidth/window.innerHeight, 0.001, 1000);
const renderer = new THREE.WebGLRenderer({antialias:true});
renderer.setSize(window.innerWidth, window.innerHeight);
document.body.appendChild(renderer.domElement);

// Controls
const controls = new THREE.OrbitControls(camera, renderer.domElement);

// Lights
const hemi = new THREE.HemisphereLight(0xffffff, 0x444444, 0.8);
scene.add(hemi);
const dir = new THREE.DirectionalLight(0xffffff, 0.6);
dir.position.set(1,2,3);
scene.add(dir);

// Lines (edges)
{
    const geom = new THREE.BufferGeometry();
    geom.setAttribute('position', new THREE.BufferAttribute(linePositions, 3));
    const mat = new THREE.LineBasicMaterial({color:0x000000});
    const lines = new THREE.LineSegments(geom, mat);
    scene.add(lines);
}

// Mesh (optional)
if ({show_mesh_js}){
    const geom2 = new THREE.BufferGeometry();
    geom2.setAttribute('position', new THREE.BufferAttribute(vertices, 3));
    geom2.setIndex(new THREE.BufferAttribute(faceIdx, 1));
    geom2.computeVertexNormals();
    const mat2 = new THREE.MeshPhongMaterial({color:0xadd8e6, transparent:true, opacity:0.9, side:THREE.DoubleSide});
    const mesh = new THREE.Mesh(geom2, mat2);
    scene.add(mesh);
}

// Camera framing
const bb = new THREE.Box3();
bb.setFromBufferAttribute(new THREE.BufferAttribute(vertices, 3));
const center = bb.getCenter(new THREE.Vector3());
const size = bb.getSize(new THREE.Vector3()).length();
camera.position.copy(center.clone().add(new THREE.Vector3(size*1.2, size*1.2, size*1.2)));
camera.lookAt(center);
controls.target.copy(center);

// Resize
window.addEventListener('resize', ()=>{camera.aspect = window.innerWidth/window.innerHeight; camera.updateProjectionMatrix(); renderer.setSize(window.innerWidth, window.innerHeight)}, false);

function animate(){requestAnimationFrame(animate); controls.update(); renderer.render(scene, camera)}
animate();
</script>
</body>
</html>"""

                # Insert arrays (avoid extremely long f-strings by replacing placeholders)
                html = html.replace('{verts_array}', str(verts_flat))
                html = html.replace('{faces_array}', str(faces_flat))
                html = html.replace('{lines_array}', str(line_pos_flat))
                html = html.replace('{show_mesh_js}', 'true' if show_mesh else 'false')

                out_path = f"{file_name}.html"
                with open(out_path, 'w', encoding='utf-8') as f:
                        f.write(html)
                logger.info(f"Three.js HTML visualization saved → {out_path}")
        except Exception as e:
                logger.error(f"Failed to write Three.js HTML: {e}", exc_info=True)
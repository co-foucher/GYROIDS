# ============================================================================
# ABAQUS SIMULATION SETUP SCRIPT - LINEAR ELASTIC UNIAXIAL LOAD CASE
#
# Reads the tet mesh produced by fTetWild (<model>.inp), applies a linear
# elastic material, grips the two end faces along a user-chosen axis and
# pushes (or pulls) on the top one with a user-chosen total force. The point
# of the load case is to make the structure's axial stiffness measurable in
# that direction.
#
# BOUNDARY CONDITIONS (roller bottom - free lateral expansion)
# ------------------------------------------------------------
#   Set-Bottom  : U<axis> = 0 on the whole bottom face slab. The face can
#                 still slide/expand in its own plane, so no platen
#                 confinement is baked into the result.
#   pin A, pin B: two individual bottom nodes, restrained in-plane only, to
#                 remove the three rigid-body modes the roller leaves free
#                 (2 transverse translations + rotation about the load axis).
#                 Without these the stiffness matrix is singular and the job
#                 dies with "numerical singularity" on the first increment.
#   Set-Top     : CLOAD along the axis, total_load / n_top_nodes per node
#                 (Abaqus applies a ConcentratedForce to *each* node of the
#                 set, so the magnitude has to be divided up front).
#
# Run by gyroid_utils.abaqus_tools.create_simulation(); every parameter
# arrives as a key=value argument after the `--` separator.
# ============================================================================

# Import Abaqus modules
import sys
from abaqus import *
from abaqusConstants import *
from caeModules import *
from driverUtils import executeOnCaeStartup
from part import *
from material import *
from section import *
from assembly import *
from step import *
from interaction import *
from load import *
from mesh import *
from optimization import *
from job import *
from sketch import *
from visualization import *
from connectorBehavior import *
import os
import logging


# ============================================================================
# parse_kv_args
# ============================================================================
def parse_kv_args(argv):
    """Parse args like key=value and return a dict."""
    out = {}
    for a in argv:
        if "=" in a:
            k, v = a.split("=", 1)
            out[k.strip()] = v.strip()
        else:
            out.setdefault("_positional", []).append(a)
    return out


# ============================================================================
# _axis_index
# ============================================================================
def _axis_index(axis):
    """'x'/'y'/'z' (any case) or '0'/'1'/'2' -> 0/1/2."""
    table = {"x": 0, "y": 1, "z": 2, "0": 0, "1": 1, "2": 2}
    key = str(axis).strip().lower()
    if key not in table:
        raise ValueError("Unknown axis %r - use x, y or z." % (axis,))
    return table[key]


# ============================================================================
# _bc_kwargs
# ============================================================================
def _bc_kwargs(fixed_dofs):
    """
    Builds the u1/u2/u3 keyword dict for a DisplacementBC: the DOF indices
    listed in `fixed_dofs` (0-based) are set to 0.0, every other DOF is left
    UNSET (i.e. free) rather than 0.0 - that distinction is the whole point
    of a roller support.
    """
    kw = {"u1": UNSET, "u2": UNSET, "u3": UNSET,
          "ur1": UNSET, "ur2": UNSET, "ur3": UNSET}
    for d in fixed_dofs:
        kw["u%d" % (d + 1)] = 0.0
    return kw

# ============================================================================
# _set_quadratic_tets
# ============================================================================
def _set_quadratic_tets(part, logger):
    """
    Reassigns every element of `part` to the 10-node quadratic tetrahedron
    (C3D10). fTetWild only ever produces 4-node linear tets, so the whole
    element set is converted in one call - no per-shape branching needed.
    """
    elements = part.elements[0:len(part.elements)]
    if len(elements) == 0:
        logger.error("_set_quadratic_tets(): part %r has no elements to convert." % (part.name,))
        raise ValueError("_set_quadratic_tets(): part %r has no elements to convert." % (part.name,))

    quad_tet = mesh.ElemType(elemCode=C3D10, elemLibrary=STANDARD,
                             secondOrderAccuracy=OFF, distortionControl=DEFAULT)
    part.setElementType(regions=(elements, ), elemTypes=(quad_tet, ))
    logger.info("mesh upgraded to quadratic tets (C3D10): %d elements." % len(elements))

# ============================================================================
# _slab_nodes
# ============================================================================
def _slab_nodes(node_array, low, high, ax, band_lo, band_hi, pad):
    """
    Returns the nodes of `node_array` lying inside the [band_lo, band_hi]
    slab along axis `ax`, spanning the full bounding box in the other two
    directions. Thin wrapper over MeshNodeArray.getByBoundingBox() that
    spares the caller the x/y/z keyword juggling.
    """
    bounds = {"xMin": low[0] - pad, "xMax": high[0] + pad,
              "yMin": low[1] - pad, "yMax": high[1] + pad,
              "zMin": low[2] - pad, "zMax": high[2] + pad}
    bounds["%sMin" % "xyz"[ax]] = band_lo
    bounds["%sMax" % "xyz"[ax]] = band_hi
    return node_array.getByBoundingBox(**bounds)


# ============================================================================
# built_static_simulation
# ============================================================================
def built_static_simulation(model_name:str,
                            working_path:str,
                            young_modulus:float,
                            poisson_ratio:float,
                            density:float, 
                            axis:str,
                            load:float,
                            compression:bool,
                            tol_frac:float,
                            num_cpus:int,
                            pin_a_coords:tuple,
                            pin_b_coords:tuple,
                            non_liner_geometry:int,
                            quadratic_tets:int):
    """
    Wrapper that owns the log file and makes sure *something* terminal is
    always written to it. abaqus_tools.create_simulation() polls that file
    for either "Simulation created successfully" or "Simulation creation
    FAILED"; if this script died silently, the caller would sit in its
    polling loop with nothing to read.
    """
    log_file = os.path.join(working_path, "generate_sim_logger_" + model_name + ".txt")
    logging.basicConfig(filename=log_file, level=logging.DEBUG,
                        format="%(asctime)s - %(levelname)s - %(message)s",
                        filemode='a')
    logger = logging.getLogger(__name__)
    try:
        _build_static_simulation(logger = logger, 
                                model_name = model_name, 
                                working_path = working_path, 
                                young_modulus = young_modulus,
                                poisson_ratio = poisson_ratio, 
                                density = density, 
                                axis = axis, 
                                load = load, 
                                compression = compression,
                                tol_frac = tol_frac, 
                                num_cpus = num_cpus, 
                                pin_a_coords = pin_a_coords, 
                                pin_b_coords = pin_b_coords,
                                non_liner_geometry = non_liner_geometry,
                                quadratic_tets = quadratic_tets)
    except Exception as exc:
        import traceback
        logger.error("traceback:\n%s" % traceback.format_exc())
        # the caller only reads the *last* line of this file, so keep the
        # terminal marker on one line whatever the exception's message looks like.
        reason = " ".join(str(exc).split())
        logger.error("Simulation creation FAILED: %s" % reason)
        sys.stdout.flush()
        raise


def _build_static_simulation(logger:logging.Logger,
                             model_name:str,
                             working_path:str,
                             young_modulus:float,
                             poisson_ratio:float,
                             density:float,
                             axis:str,
                             load:float,
                             compression:bool,
                             tol_frac:float,
                             num_cpus:int,
                             pin_a_coords:tuple,
                             pin_b_coords:tuple,
                             non_liner_geometry:int,
                             quadratic_tets:int):
    # ================================================================
    # SECTION 0 : validate the pin coordinates
    # ================================================================
    for name, coords in (("pin_a_coords", pin_a_coords), ("pin_b_coords", pin_b_coords)):
        if len(coords) != 3:
            logger.error("_build_static_simulation(): %s must be an (x, y, z) "
                         "coordinate, got %r." % (name, coords))
            raise ValueError("_build_static_simulation(): %s must be an (x, y, z) "
                             "coordinate, got %r." % (name, coords))

    # ================================================================
    # SECTION 1 : setup paths
    # ================================================================
    parent_dir = os.path.dirname(working_path)
    logger.info("Current working directory: %s" % working_path)

    ax = _axis_index(axis)
    t1, t2 = [i for i in range(3) if i != ax]
    axis_letter = "XYZ"[ax]

    # ================================================================
    # SECTION 2 : load inp file
    # ================================================================
    mesh_path = parent_dir + '/' + model_name + ".inp"
    logger.info("loading file : %s" % mesh_path)

    mdb.ModelFromInputFile(name=model_name, inputFileName=mesh_path)
    M = mdb.models[model_name]

    # ================================================================
    # SECTION 3 : get the imported part / instance
    # ================================================================
    part_name = list(M.parts.keys())[0]
    P = M.parts[part_name]
    A = M.rootAssembly
    instance_name = list(A.instances.keys())[0]
    I = A.instances[instance_name]

    session.viewports['Viewport: 1'].assemblyDisplay.setValues(
        optimizationTasks=OFF, geometricRestrictions=OFF, stopConditions=OFF)
    session.viewports['Viewport: 1'].setValues(displayedObject=P)

    # ================================================================
    # SECTION 3.1 : optionally upgrade the mesh to quadratic tets
    # ================================================================
    if quadratic_tets == 1:
        _set_quadratic_tets(P, logger)

    # ================================================================
    # SECTION 4 : apply material properties
    # ================================================================
    E, v, d = young_modulus, poisson_ratio, density

    M.Material(name='ALUMINA')
    M.materials['ALUMINA'].Density(table=((d, ), ))
    M.materials['ALUMINA'].Elastic(table=((E, v), ))
    M.HomogeneousSolidSection(material='ALUMINA', name='Section-ALUMINA', thickness=None)

    e = P.elements
    elements = e[0:len(e)]
    region = P.Set(elements=elements, name='Set-allelements')
    P.SectionAssignment(region=region,
                        sectionName='Section-ALUMINA',
                        offset=0.0,
                        offsetType=MIDDLE_SURFACE,
                        offsetField='',
                        thicknessAssignment=FROM_SECTION)
    
    logger.info("material assigned: E=%g, nu=%g, rho=%g" % (E, v, d))

    # ================================================================
    # SECTION 5 : locate the two end faces
    # ================================================================
    # Sets are built on the *assembly instance* rather than on the part so
    # that the BC/load regions below are unambiguous - part sets only reach
    # the assembly through the instance, and orphan meshes imported from an
    # .inp make that propagation easy to get wrong.
    bbox = I.nodes.getBoundingBox()
    low, high = bbox['low'], bbox['high']
    extent = high[ax] - low[ax]
    if extent <= 0:
        raise ValueError("The mesh has zero extent along %s - nothing to load."
                         % axis_letter)

    tol = tol_frac * extent
    # pad the four non-load bounds outwards so getByBoundingBox() can't drop
    # nodes sitting exactly on the box (floating point equality on a bound).
    pad = 1.0e-4 * max(high[0] - low[0], high[1] - low[1], high[2] - low[2])

    bottom_nodes = _slab_nodes(I.nodes, low, high, ax,
                               low[ax] - pad, low[ax] + tol, pad)
    top_nodes = _slab_nodes(I.nodes, low, high, ax,
                            high[ax] - tol, high[ax] + pad, pad)

    n_bot, n_top = len(bottom_nodes), len(top_nodes)
    logger.info("axis=%s, extent=%g, face tolerance=%g (%g %%)"
                % (axis_letter, extent, tol, 100.0 * tol_frac))
    logger.info("bottom face: %d nodes | top face: %d nodes" % (n_bot, n_top))
    if n_bot < 3 or n_top < 1:
        raise ValueError(
            "Face selection failed (%d bottom / %d top nodes). Increase the "
            "face tolerance, or check that the STL really has flat ends along %s."
            % (n_bot, n_top, axis_letter))

    A.Set(nodes=bottom_nodes, name='Set-Bottom')
    A.Set(nodes=top_nodes, name='Set-Top')

    # ---- the two rigid-body pins, picked as the bottom nodes closest to
    # the caller-supplied pin_a_coords / pin_b_coords ----
    def _closest_bottom_node(target_coords):
        best_label, best_sq_dist = None, None
        for n in bottom_nodes:
            nx, ny, nz = n.coordinates
            tx, ty, tz = target_coords
            sq_dist = (nx - tx) ** 2 + (ny - ty) ** 2 + (nz - tz) ** 2
            if best_sq_dist is None or sq_dist < best_sq_dist:
                best_sq_dist, best_label = sq_dist, n.label
        return best_label

    best_a = _closest_bottom_node(pin_a_coords)
    best_b = _closest_bottom_node(pin_b_coords)
    if best_a == best_b:
        logger.error(
            "_build_static_simulation(): pin_a_coords %r and pin_b_coords %r "
            "both resolved to bottom node %s - move them further apart."
            % (tuple(pin_a_coords), tuple(pin_b_coords), best_a))
        raise ValueError(
            "pin_a_coords and pin_b_coords resolved to the same bottom node "
            "(node %s) - move them further apart." % (best_a,))

    A.Set(nodes=I.nodes.sequenceFromLabels((best_a, )), name='Set-Pin-A')
    A.Set(nodes=I.nodes.sequenceFromLabels((best_b, )), name='Set-Pin-B')
    logger.info("rigid-body pins: node %s near %r (U%d, U%d) and node %s near %r (U%d)"
                % (best_a, tuple(pin_a_coords), t1 + 1, t2 + 1,
                   best_b, tuple(pin_b_coords), t2 + 1))

    # ================================================================
    # SECTION 6 : CREATE ANALYSIS STEP
    # ================================================================
    if non_liner_geometry == 1:
        NL_GEM = ON
        logger.info("non-linear geometry analysis enabled (NLGEOM=ON)")
    else:
        NL_GEM = OFF
    M.StaticStep(name='Step-Static', previous='Initial',
                 description='Linear elastic uniaxial load along %s' % axis_letter,
                 timePeriod=1.0, initialInc=1.0, minInc=1.0e-8, maxInc=1.0,
                 nlgeom=NL_GEM)

    if non_liner_geometry == 1:
        M.steps['Step-Static'].setValues(maxNumInc=1000, 
        initialInc=0.1)
    try:
        M.fieldOutputRequests['F-Output-1'].setValues(
            variables=('S', 'E', 'U', 'RF', 'CF'))
    except Exception as exc:
        logger.warning("could not override the field output request: %s" % exc)

    # History output on the two faces. Nothing reads it yet, but it makes the
    # stiffness post-processing (sum RF on the bottom / mean U on the top) a
    # pure ODB read later on, with no re-solve.
    try:
        M.HistoryOutputRequest(name='H-Bottom-RF', createStepName='Step-Static',
                               variables=('RF1', 'RF2', 'RF3'),
                               region=A.sets['Set-Bottom'], sectionPoints=DEFAULT)
        M.HistoryOutputRequest(name='H-Top-U', createStepName='Step-Static',
                               variables=('U1', 'U2', 'U3'),
                               region=A.sets['Set-Top'], sectionPoints=DEFAULT)
    except Exception as exc:
        logger.warning("could not create the history output requests: %s" % exc)

    # ================================================================
    # SECTION 7 : BOUNDARY CONDITIONS
    # ================================================================
    M.DisplacementBC(name='BC-Roller-Bottom', createStepName='Initial',
                     region=A.sets['Set-Bottom'], amplitude=UNSET,
                     distributionType=UNIFORM, fieldName='', localCsys=None,
                     **_bc_kwargs([ax]))

    M.DisplacementBC(name='BC-Pin-A', createStepName='Initial',
                     region=A.sets['Set-Pin-A'], amplitude=UNSET,
                     distributionType=UNIFORM, fieldName='', localCsys=None,
                     **_bc_kwargs([t1, t2]))

    M.DisplacementBC(name='BC-Pin-B', createStepName='Initial',
                     region=A.sets['Set-Pin-B'], amplitude=UNSET,
                     distributionType=UNIFORM, fieldName='', localCsys=None,
                     **_bc_kwargs([t2]))

    # ================================================================
    # SECTION 8 : APPLY LOAD
    # ================================================================
    # ConcentratedForce applies the given magnitude to *every* node in the
    # region, so the user's total force is split across the top face here.
    sense = -1.0 if compression else 1.0
    per_node = sense * float(load) / float(n_top)
    cf = {"cf1": 0.0, "cf2": 0.0, "cf3": 0.0}
    cf["cf%d" % (ax + 1)] = per_node

    M.ConcentratedForce(name='Load-Axial', createStepName='Step-Static',
                        region=A.sets['Set-Top'], distributionType=UNIFORM,
                        field='', localCsys=None, **cf)
    logger.info("load: total %g in %s along %s -> %g per node on %d nodes"
                % (load, "compression" if compression else "tension",
                   axis_letter, per_node, n_top))

    # ================================================================
    # SECTION 9 : CREATE JOB
    # ================================================================
    mdb.Job(name='Job-' + model_name, model=model_name, description='',
            type=ANALYSIS, atTime=None, waitMinutes=0, waitHours=0, queue=None,
            memory=90, memoryUnits=PERCENTAGE, getMemoryFromAnalysis=True,
            explicitPrecision=SINGLE, nodalOutputPrecision=SINGLE, echoPrint=OFF,
            modelPrint=OFF, contactPrint=OFF, historyPrint=OFF, userSubroutine='',
            scratch='', resultsFormat=ODB, multiprocessingMode=DEFAULT,
            numCpus=num_cpus, numDomains=num_cpus, numGPUs=0)
    mdb.jobs['Job-' + model_name].writeInput(consistencyChecking=ON)

    logger.info("job has been created : %s/Job-%s.inp" % (working_path, model_name))
    logger.info("Simulation created successfully.")
    sys.stdout.flush()


# ============================================================================
# entry point - every value arrives as a string on the Abaqus command line,
# so each one is cast here and given the same default as the GUI's widget.
# ============================================================================
if __name__ == "__main__":
    args = parse_kv_args(sys.argv[1:])
    if "input" not in args or not args["input"]:
        raise SystemExit('Missing required argument: input="..."')
    print(args.get("pin_a_coords", "0,0,0"))
    built_static_simulation(
        model_name=args["input"],
        working_path=args["output"],
        young_modulus=float(args.get("young_modulus", 300000)),
        poisson_ratio=float(args.get("poisson_ratio", 0.21)),
        density=float(args.get("density", 3.9e-09)),
        axis=args.get("axis", "z"),
        quadratic_tets=int(args.get("quadratic_tets", 0)),
        load=float(args.get("load", 1.0)),
        compression=str(args.get("compression", "1")).lower() in ("1", "true", "yes"),
        tol_frac=float(args.get("tol_frac", 0.01)),
        num_cpus=int(args.get("num_cpus", 1)),
        pin_a_coords=tuple(float(x) for x in args.get("pin_a_coords", "0,0,0").replace("'", "").replace("[", "").replace("]", "").split(",")),
        pin_b_coords=tuple(float(x) for x in args.get("pin_b_coords", "0,0,0").replace("'", "").replace("[", "").replace("]", "").split(",")),
        non_liner_geometry=int(args.get("non_linear_geometry", "0")),
    )

# ============================================================================
# ABAQUS SIMULATION SETUP SCRIPT FOR GYROID STRUCTURE
# Generates modal and steady-state dynamic analysis for aluminum gyroid
# ============================================================================

# Import Abaqus modules
import sys
from ast import main
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
import numpy as np
import os
import logging
from pathlib import Path


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


def load_gyroid_matrices(infile: str):
    """
    ============================================================================
    4) LOAD_GYROID_MATRICES
    Loads gyroid-related matrices from a NumPy .npz archive.
    ============================================================================

    The function expects the archive to contain the arrays:
    'Xperiod', 'Yperiod', 'Zperiod', and 'thickness'.

    Errors such as missing files, missing arrays, or corrupted files
    are caught and logged. In these cases, the function returns None.

    PARAMETERS
    ----------
    infile : str
        Path to the input .npz file.

    RETURNS
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] or None
        (Xperiod, Yperiod, Zperiod, thickness, gyroid_field) if successful,
        otherwise None.

    EXAMPLE
    -------
    >>> Xp, Yp, Zp, t = load_gyroid_matrices("gyroid_data.npz")
    """
    try:
        with np.load(infile) as loaded_file:
            Xres = loaded_file["Xres"]
            Yres = loaded_file["Yres"]
            Zres = loaded_file["Zres"]
            Xperiod = loaded_file["Xperiod"]
            Yperiod = loaded_file["Yperiod"]
            Zperiod = loaded_file["Zperiod"]
            thickness = loaded_file["thickness"]
            gyroid_field = loaded_file["gyroid_field"]

    except FileNotFoundError:
        logger.error(f"File not found: {infile}")
        return None
    except KeyError as e:
        logger.error(f"Missing expected array in file {infile}: {e}")
        return None
    except Exception as e:
        logger.error(f"Failed to load matrices from {infile}: {e}")
        return None
    logger.info(f"Matrices successfully loaded from: {infile}")
    return Xres, Yres, Zres, Xperiod, Yperiod, Zperiod, thickness, gyroid_field


def built_simulation_of_gyroid(model_name):
    # ================================================================
    # SECTION 1 : setup paths 
    # ================================================================
    """
    try :
        with open("temp_file.txt", "r") as f:
            model_name = f.read().strip() # .strip() is used to remove any leading/trailing whitespace or newline characters
    except FileNotFoundError:
        print("temp file not found")
        return
    except Exception as e:
        print(f"Failed to read temp file as: {e}")
        return
    Path("temp_file.txt").unlink()  # as soon as you read it, delete it
    """
    working_path = os.getcwd()
    parent_dir = os.path.dirname(working_path)

    # ================================================================
    # SECTION 1.1 : Configure logger to write to a file in the current working directory
    # ================================================================
    log_file = os.path.join(os.getcwd(), "generate_sim_logger_"+ model_name +".txt")
    logging.basicConfig(filename=log_file, level=logging.INFO,
                        format="%(asctime)s - %(levelname)s - %(message)s",
                        filemode='a')
    logger = logging.getLogger(__name__)
    logger.info(f"Current working directory: {working_path}")

    # ================================================================
    # SECTION 2 : load inp file 
    # ================================================================
    #model_name = "gyroid-z_change-6"
    mesh_path = parent_dir + "\\02 - mesh files\\" + model_name + ".inp"

    logger.info(f"loading file : {mesh_path}")

    mdb.ModelFromInputFile(name=model_name, 
        inputFileName=mesh_path)

    # ================================================================
    # SECTION 3 : scale P 
    # ================================================================
    old_part_name = list(mdb.models[model_name].parts.keys())[0]
    old_part_name_inassembly = list(mdb.models[model_name].rootAssembly.features.keys())[0] 
    part_name = old_part_name + '-scaled'


    # ==== make a copy that is scaled ====

    scale_factor = 0.1

    P = mdb.models[model_name].Part(name=part_name, 
        objectToCopy=mdb.models[model_name].parts[old_part_name], 
        compressFeatureList=ON, scale=scale_factor)

    # ==== add it to assembly ====
    A = mdb.models[model_name].rootAssembly
    P = mdb.models[model_name].parts[part_name]     #not sure if I need to redefine it...
    A.Instance(name=part_name, part=P, dependent=ON)
    I = mdb.models[model_name].rootAssembly.instances[part_name]


    # ==== delete old part ====
    del mdb.models[model_name].parts[old_part_name] # in part
    del A.features[old_part_name_inassembly] # in assembly


    # ===== view mesh =====
    session.viewports['Viewport: 1'].assemblyDisplay.setValues(
        optimizationTasks=OFF, geometricRestrictions=OFF, stopConditions=OFF)
    P = mdb.models[model_name].parts[part_name]
    session.viewports['Viewport: 1'].setValues(displayedObject=P)

    # ====== make mesh quadratic ======
    """
    elemType1 = mesh.ElemType(elemCode=C3D10, 
        elemLibrary=STANDARD, 
        secondOrderAccuracy=OFF, 
        distortionControl=DEFAULT)

    e = P.elements
    elements = e[0:len(e)]
    pickedRegions =(elements, )
    p.setElementType(regions=pickedRegions, elemTypes=(elemType1, ))
    """
    # ================================================================
    # SECTION 4 : apply material properties 
    # ================================================================
    # Define material properties for ALUMINA (aluminum oxide)
    # E: Young's modulus (MPa), v: Poisson's ratio, d: density (ton/mm³)
    E,v,d = 300000, 0.21, 3.9e-09

    mdb.models[model_name].Material(name='ALUMINA')

    mdb.models[model_name].materials['ALUMINA'].Density(table=((d, ), ))

    mdb.models[model_name].materials['ALUMINA'].Elastic(table=((E, v), ))

    mdb.models[model_name].HomogeneousSolidSection(material='ALUMINA', name=
            'Section-ALUMINA', thickness=None)

    e = P.elements
    elements = e[0:len(e)]
    region = P.Set(elements=elements, name='Set-allelements')
    P.SectionAssignment(region=region, 
        sectionName='Section-ALUMINA', 
        offset=0.0, 
        offsetType=MIDDLE_SURFACE, 
        offsetField='', 
        thicknessAssignment=FROM_SECTION)

    # ================================================================
    # SECTION 4 : CREATE ANALYSIS STEPS
    # ================================================================
    f_min = 1
    f_max = 1000000
    precision = 50  #number of points

    # ==== step 1 ====
    mdb.models[model_name].FrequencyStep(limitSavedEigenvectorRegion=None, 
        numEigen=10,
        maxEigen=f_max, 
        minEigen=f_min, 
        name='Step-Modal', 
        previous='Initial')
    #automatic field output is goode enough


    # ================================================================
    # SECTION 5 : APPLY LOAD
    # ================================================================

    #===============================================================
    # SECTION 6 : CREATE JOB
    # ================================================================

    # ====== create job ======
    mdb.Job(name='Job-' + model_name, model=model_name, description='', type=ANALYSIS, 
        atTime=None, waitMinutes=0, waitHours=0, queue=None, memory=90, 
        memoryUnits=PERCENTAGE, getMemoryFromAnalysis=True, 
        explicitPrecision=SINGLE, nodalOutputPrecision=SINGLE, echoPrint=OFF, 
        modelPrint=OFF, contactPrint=OFF, historyPrint=OFF, userSubroutine='', 
        scratch='', resultsFormat=ODB, multiprocessingMode=DEFAULT, numCpus=14, 
        numDomains=14, numGPUs=0)
    mdb.jobs['Job-' + model_name].writeInput(consistencyChecking=ON)

    logger.info(f"job has been created : {working_path}/Job-{model_name}.inp")
    logger.info(f"Run it through : python abq_windows debug=DEBUG cpus=6 job=Job-{model_name}")
    logger.info(f"Simulation created successfully.")
    sys.stdout.flush()


if __name__ == "__main__":
    args = parse_kv_args(sys.argv[1:])
    if "input" not in args or not args["input"]:
        raise SystemExit('Missing required argument: input="..."')
    model_name = args["input"]
    built_simulation_of_gyroid(model_name)

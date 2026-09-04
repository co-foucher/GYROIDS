from pathlib import Path
import time
import subprocess
import shutil
from .logger import logger
import numpy as np

"""
#=====================================================================================================================
0 - (reserved)
1 - create_simulation
2 - run_simulation
3 - _is_simulation_started
4 - wait_for_simulation_completed
5 - extract_field_at_nodeset
#=====================================================================================================================
"""

# =====================================================================
# 1) create_simulation
# =====================================================================
def create_simulation(input_path:str,
                      output_path:str,
                      file_name:str,
                      script_name:str = "generate_frequency_sim.py",
                      max_wait_time:int = 600, 
                      young_modulus:float = 300000.0,
                      poisson_ratio:float = 0.21,
                      density:float = 3.9e-09,
                      quadratic_tets:int = 0,
                      extra_args:dict = None) -> bool:
    """
    ============================================================================
    1) CREATE_SIMULATION
    Utility function to create an Abaqus simulation by invoking the appropriate
    mesh file, output folder, and Abaqus script.
    ============================================================================

    PARAMETERS
    ----------
    input_path : str
        Path to the folder containing the input mesh file (should be an inp file).
    output_path : str
        Path to the folder where the simulation input file will be written
        (again as an inp file).
    file_name : str
        Base name used to locate the input INP file and name the output files.
    script_name : str, optional
        Name of the Abaqus script to run (without .py extension). This script
        should be located in output_path and should be designed to read the
        mesh file specified by file_name and create the appropriate simulation
        input files. Adjust this if you have different scripts for different
        simulation types. Default = "generate_frequency_sim.py".
    max_wait_time : int, optional
        Maximum time to wait for simulation creation, in seconds (default = 600).
    extra_args : dict, optional
        Additional key=value arguments forwarded verbatim to the Abaqus
        script on the command line, on top of the material properties that
        every script takes. This is how simulation-type-specific options get
        through without every script having to accept every other script's
        parameters - e.g. generate_static_sim.py needs
        ``{"axis": "z", "load": 100.0, "compression": 1, "tol_frac": 0.01}``
        while generate_frequency_sim.py needs none of them. Keys and values
        must be str()-able and must not contain "=" or spaces.

    RETURNS
    -------
    success : bool
        True if the inp file was created successfully, False otherwise
        (also writes output files to disk).

    NOTES
    -----
    - The Python script used to create the simulation must be located in
      output_path.
    - Behavior: runs Abaqus in noGUI mode to execute the chosen script in that
      folder, then waits for the external script to write a log file
      'generate_sim_logger.txt' and polls that file until a
      'Simulation created successfully' message is found in its last line.
    """
    # Compose path to the input .inp (kept for compatibility with other code)
    input_inp = Path(input_path + file_name + '.inp')
    # folder where generate_sim.py lives and where we'll write temp files
    script_folder = Path(output_path)
    # Choose which Abaqus wrapper script to run based on requested simulation type
    try:
        with open(script_folder / script_name, "r") as f:
            content = f.read()
    except:
        # Protect against invalid usage
        logger.error(f"Unknown script name: {script_name}")
        return False

    # --- run abaqus headless from that folder ---
    # Running with `cwd=str(script_folder)` ensures Abaqus starts in the folder containing temp_file.txt
    cmd = ["abaqus", "cae",
           "noGUI=" + script_name,
           "--", "input=" + file_name,
           "--", "output=" + output_path,
           "--", "young_modulus=" + str(young_modulus),
           "--", "poisson_ratio=" + str(poisson_ratio),
           "--", "density=" + str(density),
           "--", "quadratic_tets=" + str(quadratic_tets)
           ]  # pass the file name as an argument to the script
    # simulation-type-specific options (axis, load, ... for the static case)
    for key, value in (extra_args or {}).items():
        cmd += ["--", f"{key}={value}"]
    logger.debug(f"Running command: {' '.join(cmd)} in {script_folder}")

    # The Abaqus scripts open their log with filemode='a', so a stale log from
    # a previous run would still end in "Simulation created successfully" and
    # the poll below would return True before this run wrote anything at all.
    temp = Path(output_path) / ("generate_sim_logger_" + file_name + ".txt")
    temp.unlink(missing_ok=True)

    subprocess.run(cmd, check=True, cwd=str(script_folder), shell=True)

    # --- wait for external script to signal completion ---
    # The external script writes 'generate_sim_logger_<file_name>.txt' and ends
    # it with either 'Simulation created successfully' or 'Simulation creation
    # FAILED: <reason>'. We poll that file until one of the two shows up.
    #
    # NOTE: the timeout is checked at the top of the loop, not only on the
    # file-not-found path. A script that starts, writes a few log lines and
    # then dies (or hits a case it can't handle - e.g. the static script
    # finding an empty face node set) leaves a log file that exists but never
    # reaches a terminal line; guarding only the missing-file branch would
    # spin here forever, and this runs in a background job thread that nothing
    # can cancel.
    start_time = time.time()
    while True:
        if time.time() - start_time > max_wait_time:
            logger.warning(f"Simulation creation did not complete within {max_wait_time} seconds. Giving up")
            return False
        try:
            with open(temp) as file:
                lines = [line.rstrip() for line in file if line.strip()]
        except FileNotFoundError:
            # Log file not present yet; wait and retry
            logger.debug("file not found, waiting...")
            time.sleep(5)
            continue

        if not lines:
            time.sleep(1)
            continue

        last_line = lines[-1]
        if "Simulation created successfully" in last_line:
            logger.info("Simulation created successfully.")
            break
        if "Simulation creation FAILED" in last_line:
            logger.error(f"The Abaqus script reported a failure: {last_line}")
            return False
        # Not ready yet: sleep briefly and try again
        logger.debug(f"simulation not created yet, waiting... last line: {last_line}")
        time.sleep(1)

    # --- delete temporary file (best-effort) ---
    # Use missing_ok=True so we don't raise if the file was removed elsewhere
    #temp_path.unlink(missing_ok=True)
    temp_path = script_folder / "abaqus.rpy"
    temp_path.unlink(missing_ok=True)
    return True


# =====================================================================
# 2) run_simulation
# =====================================================================

def run_simulation(input_path,
                   output_path,
                   file_name,
                   max_wait_time=600) -> bool:
    """
    ============================================================================
    2) RUN_SIMULATION
    Utility function to run an Abaqus simulation by invoking the appropriate
    input file and output folder. It waits for the simulation to start by
    polling for the ODB file, then returns.
    ============================================================================

    PARAMETERS
    ----------
    input_path : str
        Path to the folder containing the input INP file (kept for interface
        compatibility).
    output_path : str
        Path to the folder where the simulation will be run and where output
        files will be written.
    file_name : str
        Base name used to locate the input INP file and name the output files.
    max_wait_time : int, optional
        Maximum time to wait for the simulation to start, in seconds
        (default = 600).

    RETURNS
    -------
    success : bool
        True if no error, False otherwise.
    """
    src = Path(input_path) / ("Job-" + file_name + ".inp")
    dst = Path(output_path) / ("Job-" + file_name + ".inp")
    try:
        shutil.copyfile(src, dst)
    except (FileNotFoundError, FileExistsError) as e:
        logger.error(f"Error copying input file: {e}")
        return False

    # --- run abaqus headless from that folder ---
    cmd = ["abaqus", "job=Job-" + file_name]
    try:
        subprocess.run(cmd, check=True, cwd=str(output_path), shell=True)
    except Exception as e:
        logger.error(f"Error running Abaqus simulation for {file_name}: {e}")
        return False

    # --- Wait for the simulation to start by polling the ODB folder for the .odb file ---
    is_started = _is_simulation_started(output_path, file_name,max_wait_time=max_wait_time)  # wait up to 10 minutes for the simulation to start
    if not is_started:
        logger.warning(f"Simulation did not start properly for {file_name}")
        return False
    dst.unlink(missing_ok=True)
    return True

# =====================================================================
# 3) _is_simulation_started
# =====================================================================
def _is_simulation_started(ODB_path, file_name, max_wait_time=300)->bool:
    """
    ============================================================================
    3) _IS_SIMULATION_STARTED
    Waits for the simulation to start by polling the ODB folder for the .odb
    file.
    ============================================================================

    PARAMETERS
    ----------
    ODB_path : str
        Path to the folder where the ODB file is expected to appear.
    file_name : str
        Base name used to build the expected ODB file name.
    max_wait_time : int, optional
        Maximum time to wait for the ODB file to appear, in seconds
        (default = 300).

    RETURNS
    -------
    started : bool
        True if the ODB file was found within max_wait_time, False otherwise.
    """
    odb_file = Path(ODB_path) / ("Job-" + file_name + ".odb")
    start_time = time.time()
    while not odb_file.exists():
        logger.info("Simulation not started yet, waiting...")
        time.sleep(30)  # wait before checking again
        if time.time() - start_time > max_wait_time:
            logger.warning(f"Simulation did not start within {max_wait_time} seconds.")
            return False
    logger.info("Simulation started, ODB file found.")
    return True


# =====================================================================
# 4) wait_for_simulation_completed
# =====================================================================
def wait_for_simulation_completed(ODB_path:str,
                                  file_name:str,
                                  max_wait_time:int=300) -> bool:
    """
    ============================================================================
    4) WAIT_FOR_SIMULATION_COMPLETED
    Utility function to wait for an Abaqus simulation to complete by polling
    the log file for specific key words indicating completion or abortion.
    ============================================================================

    PARAMETERS
    ----------
    ODB_path : str
        Path to the folder containing the ODB file (output of Abaqus simulation).
    file_name : str
        Base name used to locate the log file.
    max_wait_time : int, optional
        Maximum time to wait for simulation completion, in seconds
        (default = 300, i.e. 5 minutes).

    RETURNS
    -------
    success : bool
        True if no error, False otherwise.
    """
    log_file = Path(ODB_path) / ("Job-" + file_name + ".log")
    simulation_finished = False
    start_time = time.time()
    while not simulation_finished:
        if time.time() - start_time > max_wait_time:
            logger.warning(f"Simulation did not complete within {max_wait_time} seconds.")
            return False
        try:
            with open(log_file) as file:
                lines = [line.rstrip() for line in file]
            if not lines:
                time.sleep(1)
                continue
            # Check the last line against known terminal states. Use substring
            # matching against the line itself (not `in lines`, which only
            # matches a line that is *exactly* equal to the keyword and would
            # never catch e.g. "Abaqus/Analysis exited with errors").
            last_line = lines[-1]
            if "COMPLETED" in last_line:
                logger.info("Simulation run completed.")
                return True
            elif "ABORTED" in last_line:
                logger.info("Simulation run aborted.")
                return False
            elif "exited with error" in last_line.lower():
                logger.info("Simulation run encountered an error.")
                return False
            else:
                # Not ready yet: sleep briefly and try again
                time.sleep(1)
                logger.info("simulation not completed yet, waiting...")
                logger.info(f"last line is {last_line}")
        except :
            # Log file not present yet; wait and retry
            logger.info("file not found, waiting...")
            time.sleep(1)


# =====================================================================
# 5) extract_field_at_nodeset
# =====================================================================
def extract_field_at_nodeset(odb_path:str,
                            nset_name:str,
                            field_name:str,
                            out_path:str = None) -> bool:
    """
    ============================================================================
    5) EXTRACT_FIELD_AT_NODESET
    Utility function to extract a field output at a node set, at the last
    frame of the last step, from a completed Abaqus simulation. Runs the
    pybaqus/extract_field_at_nodeset.py script through Abaqus's own Python
    interpreter (abaqus python), since odbAccess is only importable there.
    ============================================================================

    PARAMETERS
    ----------
    odb_path : str
        Full path to the ODB file to read.
    nset_name : str
        Name of the node set to extract the field output from, as defined
        in the ODB assembly (case-sensitive, usually uppercase).
    field_name : str
        Name of the field output to extract (e.g. "U", "RF", "S").
    out_path : str, optional
        Path of the text file to write the extracted values to. Defaults
        to "<odb folder>/<odb stem>_<nset_name>_<field_name>.csv".

    RETURNS
    -------
    success : bool
        True if the output file was written successfully, False otherwise.

    NOTES
    -----
    - The pybaqus script always exits with code 0, even when it fails
      internally (e.g. node set or field not found), because it only
      prints an error and returns False without setting the process exit
      code. Success is therefore checked by confirming the output file
      exists after the call, not by the subprocess return code.
    """
    odb_file = Path(odb_path)
    if not odb_file.exists():
        logger.error(f"ODB file not found: {odb_file}")
        return False

    if out_path is None:
        out_path = str(odb_file.parent / f"{odb_file.stem}_{nset_name}_{field_name}.csv")

    script_path = Path(__file__).parent / "pybaqus" / "extract_field_at_nodeset.py"

    # Remove any stale output from a previous extraction first. Otherwise a run
    # that fails inside Abaqus (bad field/component name, missing node set, ...)
    # still exits 0 - see the NOTES above - and the exists() check below would
    # find the leftover file from an earlier, different extraction and report
    # success with the wrong data still sitting at out_path.
    Path(out_path).unlink(missing_ok=True)

    # --- run the extraction script through Abaqus's own Python interpreter ---
    cmd = ["abaqus", "python", str(script_path),
           str(odb_file), nset_name, field_name, out_path]
    logger.debug(f"Running command: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True, cwd=str(odb_file.parent), shell=True)
    except Exception as e:
        logger.error(f"Error running Abaqus extraction script for {odb_file.name}: {e}")
        return False

    if not Path(out_path).exists():
        logger.error(f"Extraction did not produce an output file: {out_path}")
        return False

    logger.info(f"Field '{field_name}' at node set '{nset_name}' extracted to {out_path}")
    return True

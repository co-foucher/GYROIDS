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
3 - _wait_for_simulation_start   
4 - wait_for_simulation_completed
#=====================================================================================================================
"""

# =====================================================================
# 1) create_simulation
# =====================================================================
def create_simulation(input_path:str, 
                      output_path:str, 
                      file_name:str, 
                      script_name:str = "generate_frequency_sim.py"):
    """Create simulation input by invoking the appropriate Abaqus script."""
    """
    Parameters:
        input_path (str): path to input files (kept for interface compatibility)
        output_path (str): path where generate_sim.py lives and where temp files are written
        file_name (str): base name used to create temp_file.txt so the Abaqus script can read it
        script_name (str): name of the Abaqus script to run (without .py extension).

    NECESSARY !!!!
        !!!!! the python script to create the simulation must be located in output_path !!!!!

    Behavior:
        - writes a small temp file named 'temp_file.txt' into INP_path with the file_name NOT ANYMORE (the Abaqus script will read this to know which mesh to load)
        - runs Abaqus in noGUI mode to execute the chosen script in that folder
        - waits for the external script to write a log file 'generate_sim_logger.txt' and
          polls that file until a 'Simulation created successfully' message is found in its last line
        - deletes the temp_file.txt afterwards (best-effort)
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
        raise ValueError(f"Unknown script name: {script_name}")

    # --- write a small temp file that the Abaqus script will read ---
    # temp_file.txt contains the base file name so the Abaqus-side script knows which mesh to load
    """
    temp_path = script_folder / "temp_file.txt"
    temp_path.parent.mkdir(parents=True, exist_ok=True)  # ensure folder exists

    #try to see if a temp file already exists, if so wait a bit before checking again
    while temp_path.exists():
        logger.info("temp file already exists, waiting...")
        time.sleep(np.random.uniform(10, 30))   # wait a random time before checking again to avoid conflicts in parallel runs
    with open(temp_path, "w") as f:
        f.write(f"{file_name}")
        #note: no need to delete it as the abaqus python script does that after reading it
        #in parallel runs this is safer as having several script writting in the same file would be a problem...
    """
    # --- run abaqus headless from that folder ---
    # Running with `cwd=str(script_folder)` ensures Abaqus starts in the folder containing temp_file.txt
    cmd = ["abaqus", "cae", 
           "noGUI=" + script_name,
           "--", "input=" + file_name]  # pass the file name as an argument to the script
    
    subprocess.run(cmd, check=True, cwd=str(script_folder), shell=True)

    # --- wait for external script to signal completion ---
    # The external script is expected to write 'generate_sim_logger.txt' and append a line
    # containing 'Simulation created successfully' when done. We poll that file until we see it.
    simulation_created = False
    temp = Path(output_path) / ("generate_sim_logger_"+ file_name +".txt")
    while not simulation_created:
        try:
            with open(temp) as file:
                lines = [line.rstrip() for line in file]
            # If the last line indicates success, we're done
            if "Simulation created successfully" in lines[-1]:
                logger.info("Simulation created successfully.")
                break
            else:
                # Not ready yet: sleep briefly and try again
                time.sleep(1)
                logger.info("simulation not created yet, waiting...")
                logger.info(f"last 2 lines are {lines[-2]}")
                logger.info(f"                 {lines[-1]}")
        except FileNotFoundError:
            # Log file not present yet; wait and retry
            logger.info("file not found, waiting...")
            time.sleep(1)
    # --- delete temporary file (best-effort) ---
    # Use missing_ok=True so we don't raise if the file was removed elsewhere
    #temp_path.unlink(missing_ok=True)  
    temp_path = script_folder / "abaqus.rpy"
    temp_path.unlink(missing_ok=True)  


# =====================================================================
# 2) run_simulation
# =====================================================================

def run_simulation(input_path, 
                   output_path, 
                   file_name) -> bool:
    """Run the simulation by invoking the appropriate Abaqus input file.
    returns True if no error, False otherwise."""
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
    _wait_for_simulation_start(output_path, file_name,max_wait_time=300)  # wait up to 5 minutes for the simulation to start
    dst.unlink(missing_ok=True)  
    return True

# =====================================================================
# 3) _wait_for_simulation_start
# =====================================================================
def _wait_for_simulation_start(ODB_path, file_name, max_wait_time=300):
    """Wait for the simulation to start by polling the ODB folder for the .odb file."""
    odb_file = Path(ODB_path) / ("Job-" + file_name + ".odb")
    start_time = time.time()    
    while not odb_file.exists():
        logger.info("Simulation not started yet, waiting...")
        time.sleep(30)  # wait before checking again
        if time.time() - start_time > max_wait_time:
            logger.warning(f"Simulation did not start within {max_wait_time} seconds.")
            break
    logger.info("Simulation started, ODB file found.")


# =====================================================================
# 4) wait_for_simulation_completed
# =====================================================================
def wait_for_simulation_completed(ODB_path, file_name):
    """Wait for the simulation to complete by looking at the log file for specific key words."""
    log_file = Path(ODB_path) / ("Job-" + file_name + ".log")
    simulation_finished = False
    while not simulation_finished:
        try:
            with open(log_file) as file:
                lines = [line.rstrip() for line in file]
            # If the last line indicates success, we're done
            if "COMPLETED" in lines[-1]:
                logger.info("Simulation run completed.")
                break
            elif "ABORTED" in lines:
                logger.info("Simulation run aborted.")
                break
            else:
                # Not ready yet: sleep briefly and try again
                time.sleep(1)
                logger.info("simulation not completed yet, waiting...")
                logger.info(f"last line is {lines[-1]}")
        except :
            # Log file not present yet; wait and retry
            logger.info("file not found, waiting...")
            time.sleep(1)
# NOT USING SCHEDULE.PY AS IT IS LESS RELIABLE COMPARED TO TASK SCHEDULER AND WOULD REQUIRE THE
# SCRIPT TO BE RUNNING ALL THE TIME

# import schedule  # for scheduling tasks
# import time  # for time related tasks like sleep
# import subprocess  # for running other Python scripts
# import logging
# import sys


# # Configure logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# logger = logging.getLogger(__name__)

# python_exe = sys.executable  # Uses the current venv's Python

# def job():
#     try:
#         #check = true will raise an exception if the main script fails
#         subprocess.run([python_exe, "main.py"], check=True)
#         logger.info("Upload job completed successfully.")
#     except subprocess.CalledProcessError as e:
#         logger.error(f"Upload failed: {e}")

# schedule.every().day.at("21:00").do(job)

# while True:
#     schedule.run_pending()
#     time.sleep(60)  # pauses the script for 60 seconds to reduce CPU usage

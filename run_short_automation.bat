@echo off
REM Turn off command echoing

echo Starting YouTube Shorts Automation...
REM Print a message indicating the start of the automation

call "D:\youtube-shorts-automation\venv\Scripts\activate.bat"
REM Activate the Python virtual environment

set PYTHONPATH="D:\youtube-shorts-automation"
REM Set the PYTHONPATH environment variable to the project directory for local imports

cd "D:\youtube-shorts-automation"
REM Change the current directory to the project directory

python main.py
REM Run the main Python script

echo Script execution completed.
REM Print a message indicating the completion of the script execution

pause
REM Pause the script to keep the command window open

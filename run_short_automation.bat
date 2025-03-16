@echo off
echo Starting YouTube Shorts Automation...
call "D:\youtube-shorts-automation\venv\Scripts\activate.bat"
set PYTHONPATH="D:\youtube-shorts-automation"
cd "D:\youtube-shorts-automation"
python main.py
echo Script execution completed.
pause

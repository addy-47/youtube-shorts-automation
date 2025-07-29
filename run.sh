#!/bin/bash

echo "[$(date)] Starting YouTube Shorts Automation..." >> /home/addy/projects/youtube-shorts-automation/cron.log

cd /home/addy/projects/youtube-shorts-automation || exit 1

# Activate the Python virtual environment using absolute path
source /home/addy/projects/youtube-shorts-automation/.venv/bin/activate

# Set PYTHONPATH
export PYTHONPATH="/home/addy/projects/youtube-shorts-automation"

# Git operations
/usr/bin/git add .
/usr/bin/git commit -m "Automated commit before running script"
/usr/bin/git checkout master

# Use full path for switch-addy, or wrap it in a full command (see tip below)
source /home/addy/.bashrc
switch-addy

# Run main.py using full python path
/home/addy/projects/youtube-shorts-automation/.venv/bin/python /home/addy/projects/youtube-shorts-automation/main.py

echo "[$(date)] Script execution completed." >> /home/addy/projects/youtube-shorts-automation/cron.log

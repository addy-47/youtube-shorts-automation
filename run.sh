#!/bin/bash
export PATH="/usr/local/bin:/usr/bin:/bin:/home/addy/.local/bin:$PATH"
export HOME="/home/addy"
LOG_FILE="/home/addy/projects/youtube-shorts-automation/cron.log"

echo "----------------------------------------" >> "$LOG_FILE"
echo "[$(date)] Starting YouTube Shorts Automation..." >> "$LOG_FILE"

cd /home/addy/projects/youtube-shorts-automation >> "$LOG_FILE" 2>&1 || { echo "[$(date)] Failed to change directory." >> "$LOG_FILE"; exit 1; }
echo "[$(date)] Changed directory to $(pwd)" >> "$LOG_FILE"

source /home/addy/projects/youtube-shorts-automation/.venv/bin/activate >> "$LOG_FILE" 2>&1 || { echo "[$(date)] Failed to activate virtual environment." >> "$LOG_FILE"; exit 1; }
echo "[$(date)] Activated virtual environment" >> "$LOG_FILE"

export PYTHONPATH="/home/addy/projects/youtube-shorts-automation"
echo "[$(date)] Set PYTHONPATH" >> "$LOG_FILE"

/usr/bin/git add . >> "$LOG_FILE" 2>&1
echo "[$(date)] Staged changes" >> "$LOG_FILE"

/usr/bin/git commit --allow-empty -m "Automated commit before running script" >> "$LOG_FILE" 2>&1
echo "[$(date)] Committed changes" >> "$LOG_FILE"

/usr/bin/git checkout master >> "$LOG_FILE" 2>&1
echo "[$(date)] Checked out master branch" >> "$LOG_FILE"

gcloud config set account adhbutg@gmail.com >> "$LOG_FILE" 2>&1
echo "[$(date)] Set gcloud account" >> "$LOG_FILE"

/home/addy/projects/youtube-shorts-automation/.venv/bin/python /home/addy/projects/youtube-shorts-automation/main.py >> "$LOG_FILE" 2>&1
echo "[$(date)] Ran Python script" >> "$LOG_FILE"

echo "[$(date)] Script execution completed." >> "$LOG_FILE"
echo "----------------------------------------" >> "$LOG_FILE"
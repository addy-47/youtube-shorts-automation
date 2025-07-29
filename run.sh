#!/bin/bash
export PATH="/usr/local/bin:/usr/bin:/bin:/home/addy/.local/bin:$PATH"
export HOME="/home/addy"
echo "[$(date)] Starting YouTube Shorts Automation..." >> /home/addy/projects/youtube-shorts-automation/cron.log
cd /home/addy/projects/youtube-shorts-automation || exit 1
source /home/addy/projects/youtube-shorts-automation/.venv/bin/activate
export PYTHONPATH="/home/addy/projects/youtube-shorts-automation"
/usr/bin/git add .
/usr/bin/git commit -m "Automated commit before running script"
/usr/bin/git checkout master
source /home/addy/.bashrc
source /home/addy/projects/scripts/gcp-k8s/gcloud-kubectl-switch.sh && switch-addy
/home/addy/projects/youtube-shorts-automation/.venv/bin/python /home/addy/projects/youtube-shorts-automation/main.py
echo "[$(date)] Script execution completed." >> /home/addy/projects/youtube-shorts-automation/cron.log
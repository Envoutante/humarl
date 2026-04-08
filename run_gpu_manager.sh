#!/bin/bash
# Wrapper script to run GPU manager with PTY for unbuffered output

LOG_FILE="${1:-gpu_manager.log}"
shift

# Use script to create PTY for unbuffered output
script -q -f -c "conda run -n hyr2pymarl python3 -u src/gpu_manager.py $*" "$LOG_FILE" &

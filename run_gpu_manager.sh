#!/bin/bash
# Wrapper script to run GPU manager with PTY for unbuffered output

set -euo pipefail

LOG_FILE="gpu_manager.log"
if [[ $# -gt 0 && "$1" != --* ]]; then
	LOG_FILE="$1"
	shift
fi

CONDA_BASE="$(conda info --base)"
INNER_CMD="source \"$CONDA_BASE/etc/profile.d/conda.sh\" && conda activate hyr2pymarl && python3 -u src/gpu_manager.py $*"

# Use script to create PTY for unbuffered output
script -q -f -c "bash -lc '$INNER_CMD'" "$LOG_FILE" &

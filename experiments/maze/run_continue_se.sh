#!/bin/bash

# To continue fill in EXPERIMENT_DIR
EXPERIMENT_DIR=""

source ../../venv/bin/activate
python run_se_simple_maze_pytorch.py \
    --viz-port=8097 \
    --start-count=800 \
    --experiment-dir="$EXPERIMENT_DIR"\
    --jobid 0


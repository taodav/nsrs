#!/bin/bash

EXPERIMENT_DIR=""

source ../../venv/bin/activate
python run_se_maze.py \
    --viz-port=8097 \
    --start-count=800 \
    --experiment-dir="$EXPERIMENT_DIR"\
    --jobid 0

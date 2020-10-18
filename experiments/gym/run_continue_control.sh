#!/bin/bash

# TO continue experiments. Fill in EXPERIMENT_DIR
EXPERIMENT_DIR=""

source ../../venv/bin/activate
python run_se_control.py \
    --viz-port=8097 \
    --start-count=1900 \
    --experiment-dir="$EXPERIMENT_DIR"\
    --jobid 0

#!/bin/bash

source ../../venv/bin/activate
python run_se_control.py \
    --viz-port=8097 \
    --epochs=1 \
    --learning-rate=0.00025 \
    --iters-per-update=50000 \
    --batch-size=64 \
    --update-frequency=3 \
    --steps-per-epoch=3000 \
    --dropout-p=0.1 \
    --exp-priority=0.0 \
    --discount=0.8 \
    --learn-representation=true \
    --start-count=0 \
    --train-nstep=1 \
    --depth=5 \
    --train-reward=false \
    --reward-learning='combined' \
    --deterministic \
    --env='acrobot' \
    --reward-type='novelty_reward' \
    --action-type='d_step_q_planning' \
    --score-func='avg_knn_scores' \
    --higher-dim-obs=true \
    --monitor=true \
    --obs-per-state=4 \
    --knn='batch_knn' \
    --offline-plotting=true \
    --description='lower learning rate' \
    --jobid 0



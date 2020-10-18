#!/bin/bash

source ../../venv/bin/activate
python run_se_maze.py \
    --viz-port=8097 \
    --epochs=1 \
    --learning-rate=0.000025 \
    --iters-per-update=30000 \
    --update-frequency=1 \
    --batch-size=64 \
    --steps-per-epoch=1000 \
    --size-maze=15 \
    --dropout-p=0.1 \
    --beta=0.0 \
    --exp-priority=0.0 \
    --discount=0.8 \
    --maze-walls=false \
    --learn-representation=true \
    --start-count=0 \
    --train-nstep=1 \
    --internal-dim=3 \
    --depth=5 \
    --reward-type='novelty_reward' \
    --action-type='d_step_q_planning' \
    --score-func='ranked_avg_knn_scores' \
    --encoder-prop-td=false \
    --rnn-q-func=false \
    --knn='batch_knn' \
    --offline-plotting=false \
    --description='Fixed edge case in knn scores' \
    --jobid 0

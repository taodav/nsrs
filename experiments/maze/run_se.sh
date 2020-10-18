#!/bin/bash

source ../../venv/bin/activate
python run_se_simple_maze_pytorch.py \
    --viz-port=8097 \
    --epochs=1 \
    --learning-rate=0.0001 \
    --iters-per-update=30000 \
    --update-frequency=3 \
    --batch-size=64 \
    --steps-per-epoch=1000 \
    --size-maze=21 \
    --dropout-p=0.1 \
    --exp-priority=0.0 \
    --discount=0.8 \
    --maze-walls=true \
    --learn-representation=true \
    --epsilon-start=0.2 \
    --epsilon-min=0.2 \
    --start-count=0 \
    --train-nstep=1 \
    --depth=5 \
    --reward-type='novelty_reward' \
    --action-type='d_step_q_planning' \
    --score-func='avg_knn_scores' \
    --encoder-prop-td=false \
    --rnn-q-func=false \
    --knn='batch_knn' \
    --offline-plotting=true \
    --description='Fixed double q loss function' \
    --jobid 0


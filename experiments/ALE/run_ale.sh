#!/bin/bash
# Be sure to have your environment active when you run this.

source ../../venv/bin/activate
python run_se_ALE.py \
    --viz-port=8097 \
    --epochs=1 \
    --learning-rate=0.0001 \
    --iters-per-update=30000 \
    --batch-size=128 \
    --update-frequency=10 \
    --steps-per-epoch=25000 \
    --dropout-p=0.1 \
    --exp-priority=0.0 \
    --discount=0.8 \
    --learn-representation=true \
    --start-count=0 \
    --train-nstep=1 \
    --depth=5 \
    --slack-ratio=10 \
    --internal-dim=5 \
    --train-reward=true \
    --reward-learning='combined' \
    --deterministic \
    --env='montezumas revenge' \
    --reward-type='novelty_reward' \
    --action-type='d_step_q_planning' \
    --score-func='avg_knn_scores' \
    --higher-dim-obs=true \
    --monitor=false \
    --obs-per-state=1 \
    --knn='batch_knn' \
    --offline-plotting=true \
    --description='Normalized inference transition loss, more iterations per update' \
    --jobid 0



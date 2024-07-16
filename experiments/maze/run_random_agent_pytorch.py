"""
Random agent baseline.

We use this to see the baseline for our exploration metrics, ie.

- Number of new states per x steps
- set of states / total steps
"""

import os
import sys
import logging
import numpy as np
import joblib
import matplotlib.pyplot as plt
from joblib import hash, dump

sys.path.append(os.getcwd())

from nsrl.default_parser import process_se_args
from nsrl.agent import SEAgent
from nsrl.learning_algos.NSRS_pytorch import NSRS
from simple_maze_env_pytorch import MyEnv as simple_maze_env
import nsrl.experiment.base_controllers as bc
from nsrl.experiment.exploration_helpers import ExplorationMetricController
from nsrl.helper.plot import Plotter
from definitions import ROOT_DIR

from datetime import datetime

from nsrl.policies import RandomPolicy

class Defaults:
    # ----------------------
    # Experiment Parameters
    # ----------------------
    STEPS_PER_EPOCH = 1000
    EPOCHS = 5
    STEPS_PER_TEST = 1000
    PERIOD_BTW_SUMMARY_PERFS = 1

    # ----------------------
    # Environment Parameters
    # ----------------------
    FRAME_SKIP = 1

    # ----------------------
    # DQN Agent parameters:
    # ----------------------
    UPDATE_RULE = 'rmsprop'
    LEARNING_RATE = 0.00005
    LEARNING_RATE_DECAY = 1
    DISCOUNT = 0.8
    DISCOUNT_INC = 0.995
    DISCOUNT_MAX = 0.8
    RMS_DECAY = 0.9
    RMS_EPSILON = 0.0001
    MOMENTUM = 0
    CLIP_NORM = 1.0
    EPSILON_START = 0.5
    EPSILON_MIN = 0.5
    EPSILON_DECAY = 100
    UPDATE_FREQUENCY = 1
    REPLAY_MEMORY_SIZE = 1000
    BATCH_SIZE = 64
    FREEZE_INTERVAL = 1000
    DETERMINISTIC = False

    LEARN_REPRESENTATION = True

    # REWARD_TYPE = 'hash_count_reward'
    # REWARD_TYPE = 'transition_loss_reward'
    # REWARD_TYPE = 'novelty_reward'
    REWARD_TYPE = 'null'

    ACTION_TYPE = 'random'

    SCORE_FUNC = 'ranked_avg_knn_scores'
    KNN = 'batch_knn'

    # For loading network/dataset
    # START_COUNT = 500
    START_COUNT = 0
    # EXPERIMENT_DIR = os.path.join(os.getcwd(), 'experiments', "maze novelty reward with d step q planning_2019-08-31 17-43-42_2631771")
    EXPERIMENT_DIR = None

    XTRA = ''
    # ----------------------
    # Representation NN parameters:
    # ----------------------

    # for LEARN_REPRESENTATION:
    DROPOUT_P = 0.1
    INTERNAL_DIM = 3
    BETA = 0.0
    CONSECUTIVE_DISTANCE = 0.5
    SLACK_RATIO = 6

    # for REWARD_TYPE == 'novelty_reward':
    K = 5

    # for ACTION_TYPE == 'd_step_q_planning':
    DEPTH = 5

    SIZE_MAZE = 21
    MAZE_WALLS = True
    HIGHER_DIM_OBS = False # Not implemented higher_dim_obs yet

    # ITERS_PER_UPDATE = 30000
    ITERS_PER_UPDATE = 1

    # For plotting
    OFFLINE_PLOTTING = True

    # Priority replay
    EXP_PRIORITY = 0.0

    TRAIN_NSTEP = 1
    ENCODER_PROP_TD = False

    RNN_Q_FUNC = False

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # --- Parse parameters ---
    parameters = process_se_args(sys.argv[1:], Defaults)

    if parameters.deterministic:
        rng = np.random.RandomState(123456)
    else:
        rng = np.random.RandomState()

    # --- Instantiate environment ---
    env = simple_maze_env(rng,
                          maze_walls=parameters.maze_walls,
                          higher_dim_obs=parameters.higher_dim_obs,
                          size_maze=parameters.size_maze,
                          intern_dim=parameters.internal_dim)

    parameters.env_name = 'random_agent_%d' % parameters.size_maze

    if not parameters.maze_walls:
        parameters.env_name = 'random_agent_wallless_%d' % parameters.size_maze

    # --- Instantiate learning_algo ---
    learning_algo = NSRS(
        env,
        rho=parameters.rms_decay,
        rms_epsilon=parameters.rms_epsilon,
        beta=parameters.beta,
        momentum=parameters.momentum,
        clip_norm=parameters.clip_norm,
        freeze_interval=parameters.freeze_interval,
        batch_size=parameters.batch_size,
        update_rule=parameters.update_rule,
        random_state=rng,
        high_int_dim=False,
        internal_dim=parameters.internal_dim,
        transition_dropout_p=parameters.dropout_p)

    test_policy = RandomPolicy(learning_algo, env.nActions(), rng)
    train_policy = RandomPolicy(learning_algo,
                                         env.nActions(),
                                         rng)


    root_save_path = os.path.join(ROOT_DIR, "examples", "test_CRAR", "experiments")
    try:
        os.mkdir(root_save_path)
    except Exception:
        pass

    h = parameters.env_name

    parameters.experiment_dir = os.path.join(root_save_path, h)

    try:
        os.mkdir(parameters.experiment_dir)
    except Exception:

        raise Exception("Experiment already exists")
    plotter = Plotter(parameters.experiment_dir,
                      env_name=parameters.env_name,
                      host=parameters.viz_host,
                      port=parameters.viz_port,
                      offline=parameters.offline_plotting)
    # --- Instantiate agent ---
    agent = SEAgent(
        env,
        learning_algo,
        plotter,
        replay_memory_size=parameters.replay_memory_size,
        replay_start_size=parameters.batch_size,
        batch_size=parameters.batch_size,
        random_state=rng,
        train_policy=train_policy,
        test_policy=test_policy,
        reset_dataset_per_epoch=True
    )

    agent.attach(bc.VerboseController(
        evaluate_on='action',
        periodicity=1))


    # This controller currently only works for fully observable environments

    agent.attach(ExplorationMetricController(
        evaluate_on='action',
        periodicity=1,
        reset_every='episode',
        env_name=parameters.env_name,
        experiment_dir=parameters.experiment_dir,
        # baseline_file=baseline_data_fname
    ))

    agent.run(parameters.epochs, parameters.steps_per_epoch)


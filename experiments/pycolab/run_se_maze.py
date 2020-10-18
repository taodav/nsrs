"""
non-simple maze (but optimized for sample efficiency)

"""
import sys
import os
import logging
import numpy as np
import json
import torch
import copy

from deer.default_parser import process_se_args, stringify_params
from deer.agent import SEAgent
from deer.learning_algos.NAR_pytorch import NAR
from maze_env import MazeEnv

import deer.experiment.base_controllers as bc
import deer.experiment.exploration_helpers as eh
from deer.helper.plot import Plotter
from deer.helper.knn import ranked_avg_knn_scores, avg_knn_scores, batch_knn, batch_count_scaled_knn
from deer.helper.data import Bunch
from definitions import ROOT_DIR

from deer.policies import EpsilonGreedyPolicy
import deer.policies.exploration_policies as ep

class Defaults:
    # ----------------------
    # Experiment Parameters
    # ----------------------
    STEPS_PER_EPOCH = 4000
    EPOCHS = 1
    STEPS_PER_TEST = 4000
    PERIOD_BTW_SUMMARY_PERFS = 1

    # ----------------------
    # Environment Parameters
    # ----------------------
    FRAME_SKIP = 2

    # ----------------------
    # DQN Agent parameters:
    # ----------------------
    UPDATE_RULE = 'rmsprop'
    LEARNING_RATE = 0.000025 # TEST THIS
    LEARNING_RATE_DECAY = 1
    DISCOUNT = 0.8
    DISCOUNT_INC = 0.995
    DISCOUNT_MAX = 0.8
    RMS_DECAY = 0.9
    RMS_EPSILON = 0.0001
    MOMENTUM = 0
    CLIP_NORM = 1.0
    EPSILON_START = 0.0
    EPSILON_MIN = 0.0
    EPSILON_DECAY = 100
    UPDATE_FREQUENCY = 1
    REPLAY_MEMORY_SIZE = STEPS_PER_EPOCH * EPOCHS
    BATCH_SIZE = 64
    FREEZE_INTERVAL = 1000
    DETERMINISTIC = True

    LEARN_REPRESENTATION = True

    # REWARD_TYPE = 'hash_count_reward'
    # REWARD_TYPE = 'transition_loss_reward'
    REWARD_TYPE = 'novelty_reward'
    # REWARD_TYPE = 'rnd'
    # REWARD_TYPE = 'null'

    # ACTION_TYPE = '1_step_q_planning'
    ACTION_TYPE = 'd_step_q_planning'
    # ACTION_TYPE = 'd_step_reward_planning'
    # ACTION_TYPE = 'q_planning'
    # ACTION_TYPE = 'reward_argmax'
    # ACTION_TYPE = 'q_argmax'
    # ACTION_TYPE = 'bootstrap_q'

    # SCORE_FUNC = 'ranked_avg_knn_scores'
    SCORE_FUNC = 'avg_knn_scores'
    KNN = 'batch_knn'

    # For loading network/dataset
    # START_COUNT = 1600
    START_COUNT = 0
    # EXPERIMENT_DIR = os.path.join(os.getcwd(), 'experiments', "N_maze null with bootstrap q_2019-09-20 12-03-37_19525597")
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
    SLACK_RATIO = 12

    # for REWARD_TYPE == 'novelty_reward':
    K = 5

    # for ACTION_TYPE == 'd_step_q_planning':
    DEPTH = 5

    SIZE_MAZE = 15 # only two options are 10 and 21
    # MAZE_WALLS = False
    HIGHER_DIM_OBS = False # Not implemented higher_dim_obs yet

    # ITERS_PER_UPDATE = 1
    ITERS_PER_UPDATE = 1

    # For plotting
    OFFLINE_PLOTTING = False

    # Priority replay
    EXP_PRIORITY = 0.0

    TRAIN_NSTEP = 1
    ENCODER_PROP_TD = False

    RNN_Q_FUNC = False
    TRAIN_CSC_DIST = True


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    parameters = process_se_args(sys.argv[1:], Defaults)

    parameters.env_name = "N_maze " + parameters.env_name

    parameters.replay_start_size = parameters.batch_size

    measure_exploration = parameters.size_maze != 15

    # clearing up clutter for printing args
    if not parameters.learn_representation:
        del parameters.dropout_p
        del parameters.internal_dim
        del parameters.beta
        del parameters.consec_dist
        del parameters.slack_ratio

    if not parameters.reward_type == 'novelty_reward':
        del parameters.k

    if not parameters.action_type in ['d_step_reward_planning', 'd_step_q_planning']:
        del parameters.depth

    if parameters.deterministic:
        # For reproducibility
        seed = int(parameters.job_id)
        np.random.seed(seed)
        torch.manual_seed(seed)

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        parameters.torch_version = torch.__version__

        rng = np.random.RandomState(seed)
    else:
        rng = np.random.RandomState()

    # --- Create experiment directory
    job_id = parameters.job_id
    testing = job_id == str(0)
    # testing=False

    root_save_path = os.path.join(ROOT_DIR, "examples", "pycolab", "experiments")
    try:
        os.mkdir(root_save_path)
    except Exception:
        pass

    # TEST THIS
    continue_running = parameters.experiment_dir is not None and parameters.start_count > 0
    if not continue_running:
        parameters.network_fname = None
        parameters.dataset_fname = None

    if continue_running:
        from deer.helper.data import Bunch

        print("Resuming training from directory %s" % parameters.experiment_dir)
        param_fname = os.path.join(parameters.experiment_dir, "parameters.json")
        with open(param_fname, 'r') as f:
            new_parameters = json.load(f)

        new_parameters = Bunch(new_parameters)
        new_parameters.network_fname = os.path.join(parameters.experiment_dir, "model.epoch=%d" % parameters.start_count)
        new_parameters.dataset_fname = os.path.join(parameters.experiment_dir, "dataset.epoch=%d.pkl" % parameters.start_count)

        new_parameters.start_count = parameters.start_count
        new_parameters.env_name += f'_{new_parameters.job_id}'
        new_parameters.job_id = parameters.job_id
        new_parameters.viz_port = parameters.viz_port
        new_parameters.viz_host = parameters.viz_host
        if testing:
            new_parameters.iters_per_update = parameters.iters_per_update
            new_parameters.offline_plotting = False
        parameters = new_parameters

        start_count = parameters.start_count

    LOG_PERIODICITY = 100

    h = parameters.env_name + '_' + parameters.job_id
    parameters.experiment_dir = os.path.join(root_save_path, h)

    os.mkdir(parameters.experiment_dir)

    # Save parameters here
    param_dict = vars(copy.deepcopy(parameters))

    params_fname = os.path.join(parameters.experiment_dir, "parameters.json")
    with open(params_fname, 'w') as f:
        json.dump(param_dict, f)

    score_func = ranked_avg_knn_scores
    if parameters.score_func == "avg_knn_scores":
        score_func = avg_knn_scores

    knn = batch_knn
    if parameters.knn == "batch_count_scaled_knn":
        knn = batch_count_scaled_knn

    parameters.score_func = score_func
    parameters.knn = knn

    internal_dim = None if not hasattr(parameters, 'internal_dim') else parameters.internal_dim

    env = MazeEnv(rng=rng,
                  size_maze=parameters.size_maze,
                  intern_dim=internal_dim)

    dataset = None
    # reload dataset here if need be

    if hasattr(parameters, 'dataset_fname') and parameters.dataset_fname is not None and continue_running:
        from deer.helper.data import DataSet

        dataset = DataSet.load(parameters.dataset_fname)
        parameters.dataset_fname = None
        env = dataset._environment

    plotter = Plotter(parameters.experiment_dir,
                      env_name=h,
                      host=parameters.viz_host,
                      port=parameters.viz_port,
                      offline=parameters.offline_plotting)

    plotter.plot_text('hyperparams', stringify_params(parameters))

    # --- Instantiate learning_algo ---
    learning_algo = NAR(
        env,
        random_state=rng,
        high_int_dim=False,
        rnd_network=parameters.reward_type == 'rnd',
        **vars(parameters))

    if parameters.action_type == 'q_argmax':
        test_policy = ep.QArgmaxPolicy(learning_algo, env.nActions(), rng, parameters.epsilon_start)
        train_policy = ep.QArgmaxPolicy(learning_algo, env.nActions(), rng, parameters.epsilon_start)
    elif parameters.action_type == 'd_step_q_planning':
        test_policy = ep.MCPolicy(learning_algo, parameters.reward_type, env.nActions(), rng, depth=parameters.depth, epsilon_start=parameters.epsilon_start)
        train_policy = ep.MCPolicy(learning_algo, parameters.reward_type, env.nActions(), rng, depth=parameters.depth, epsilon_start=parameters.epsilon_start)
    elif parameters.action_type == 'bootstrap_q':
        test_policy = ep.BootstrapDQNPolicy(learning_algo, env.nActions(), rng, parameters.epsilon_start)
        train_policy = ep.BootstrapDQNPolicy(learning_algo, env.nActions(), rng, parameters.epsilon_start)
    elif parameters.action_type == 'd_step_reward_planning':
        test_policy = ep.MCRewardPolicy(learning_algo, parameters.reward_type, env.nActions(), rng, depth=parameters.depth, epsilon_start=parameters.epsilon_start)
        train_policy = ep.MCRewardPolicy(learning_algo, parameters.reward_type, env.nActions(), rng, depth=parameters.depth, epsilon_start=parameters.epsilon_start)
    else:
        test_policy = EpsilonGreedyPolicy(learning_algo, env.nActions(), rng, parameters.epsilon_start)
        train_policy = EpsilonGreedyPolicy(learning_algo, env.nActions(), rng, parameters.epsilon_start)

    # --- Instantiate agent ---
    # We might need to change this.
    train_q = parameters.action_type != 'd_step_reward_planning'
    agent = SEAgent(
        env,
        learning_algo,
        plotter,
        random_state=rng,
        train_policy=train_policy,
        test_policy=test_policy,
        train_q=train_q,
        reload=continue_running,
        dataset=dataset,
        **vars(parameters))

    if hasattr(parameters, 'rnn_q_func') and parameters.rnn_q_func:
        agent.attach(eh.RNNQHistoryController(
            evaluate_on='action_taken',
            periodicity=1
        ))

    print("The parameters hash is: {}".format(h))
    print("The parameters are: {}".format(parameters))

    start_count = parameters.replay_start_size

    agent.attach(bc.CheckpointController(
        evaluate_on='action',
        periodicity=LOG_PERIODICITY,
        experiment_dir=parameters.experiment_dir,
        start_count=parameters.start_count,
        save_env=False
    ))

    agent.attach(bc.VerboseController(
        evaluate_on='action',
        periodicity=1,
        start_count=parameters.start_count))

    baseline_data_fname = os.path.join(os.getcwd(), 'plots', 'baselines', 'random_agent_%d.json' % parameters.size_maze)

    if not parameters.maze_walls:
        baseline_data_fname=os.path.join(os.getcwd(), 'plots', 'baselines', 'random_agent_%d.json' % parameters.size_maze)

    # Plotting controllers
    # This controller currently only works for fully observable environments
    if measure_exploration:
        agent.attach(eh.ExplorationMetricController(
            evaluate_on='action',
            periodicity=1,
            reset_every='epoch',
            baseline_file=baseline_data_fname,
            hyperparams=param_dict,
            start_count=parameters.start_count
        ))
    else:
        agent.attach(eh.UniqueStateCounterController(
            plotter,
            evaluate_on='action',
            periodicity=1
        ))
    loss_plotting_sum_over = 1000
    if testing:
        loss_plotting_sum_over = 10
    # This means that every periodicity * sum_over we plot
    agent.attach(eh.LossPlottingController(
        plotter,
        evaluate_on='train_step',
        periodicity=10,
        sum_over=loss_plotting_sum_over
    ))

    if parameters.reward_type == 'novelty_reward':
        agent.attach(eh.NoveltyRewardController(
            evaluate_on='train_loop',
            periodicity=1,
            k=parameters.k,
            score_func=score_func,
            knn=knn,
            secondary=False
        ))
    elif parameters.reward_type == 'hash_count_reward':
        agent.attach(eh.HashCountRewardController(
            evaluate_on='action',
            periodicity=1,
            input_dims=env.inputDimensions()[0],
            secondary=False,
            discrete=True
        ))
    elif parameters.reward_type == 'transition_loss_reward':
        agent.attach(eh.TransitionLossRewardController(
            evaluate_on='train_loop',
            periodicity=1,
            secondary=False
        ))
    elif parameters.reward_type == 'rnd':
        agent.attach(eh.RNDRewardController(
            evaluate_on='train_loop',
            periodicity=1,
            secondary=False
        ))

    mapping_eval = "action"
    mapping_periodicity = LOG_PERIODICITY

    abstr_plotting_periodicity = LOG_PERIODICITY
    abstr_plotting_evaluate = 'action'
    if testing:
        abstr_plotting_periodicity = 25000
        abstr_plotting_evaluate = 'train_step'


    start_count = parameters.replay_start_size
    if internal_dim is not None and internal_dim < 4:
        agent.attach(eh.AbstractRepPlottingController(
            plotter,
            evaluate_on=abstr_plotting_evaluate,
            start_count=start_count,
            periodicity=abstr_plotting_periodicity
        ))

    agent.attach(eh.MapPlottingController(
        plotter,
        evaluate_on=mapping_eval,
        periodicity=mapping_periodicity,
        k=parameters.k if hasattr(parameters, 'k') else 0,
        learn_representation=parameters.learn_representation,
        reward_type=parameters.reward_type,
        train_q=train_q,
        internal_dim=internal_dim,
        plot_quiver=False,
        start_count=parameters.start_count
    ))
    # Every epoch end, one has the possibility to modify the learning rate using a LearningRateController. Here we
    # wish to update the learning rate after every training epoch (periodicity=1), according to the parameters given.
    agent.attach(bc.LearningRateController(
        initial_learning_rate=parameters.learning_rate,
        learning_rate_decay=parameters.learning_rate_decay,
        periodicity=1))

    # Same for the discount factor.
    agent.attach(bc.DiscountFactorController(
        initial_discount_factor=parameters.discount,
        discount_factor_growth=parameters.discount_inc,
        discount_factor_max=parameters.discount_max,
        evaluate_on='action',
        periodicity=1))

    # During training epochs, we want to train the agent after every [parameters.update_frequency] action it takes.
    # Plus, we also want to display after each training episode (!= than after every training) the average bellman
    # residual and the average of the V values obtained during the last episode, hence the two last arguments.
    agent.attach(bc.TrainerController(
        evaluate_on='action',
        periodicity=parameters.update_frequency,
        show_episode_avg_V_value=True,
        show_avg_Bellman_residual=True))

    agent.run(parameters.epochs, parameters.steps_per_epoch, start_count=start_count, break_on_done=True)
    print("environment states")
    print(env._trajectory)


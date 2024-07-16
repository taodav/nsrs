"""
Acrobot-v1

"""
import sys
import os
import logging
import json
import torch
import copy
import numpy as np
from joblib import hash, dump

from nsrl.default_parser import process_gym_args, stringify_params
from nsrl.agent import SEAgent
from nsrl.learning_algos.NSRS_pytorch import NSRS
from control_env import MyEnv as Control_env
import nsrl.experiment.base_controllers as bc
import nsrl.experiment.exploration_helpers as eh
from nsrl.helper.plot import Plotter
from definitions import ROOT_DIR
from nsrl.helper.knn import ranked_avg_knn_scores, avg_knn_scores, batch_knn, batch_count_scaled_knn

from nsrl.policies import EpsilonGreedyPolicy
import nsrl.policies.exploration_policies as ep
"""
TODO: 
* Try repeated transitions (with different skulls) to see if approach can ignore noise - DONE
* find states when agent loses a life, and don't add transition.
* Find better way to correlate abstr repr and image (maybe through an index on abstr repr?) - DONE
* Separate intrinsic and extrinsic rewards
* make sure to not add transitions when env resets
"""

class Defaults:
    # ----------------------
    FRAME_SKIP = 1

    # ----------------------
    # DQN Agent parameters:
    # ----------------------
    UPDATE_RULE = 'rmsprop'
    LEARNING_RATE = 0.00025

    # Experiment Parameters
    # ----------------------
    STEPS_PER_EPOCH = 3000
    EPOCHS = 1
    STEPS_PER_TEST = 2500
    PERIOD_BTW_SUMMARY_PERFS = 1

    # ----------------------
    # Environment Parameters
    # ----------------------
    LEARNING_RATE_DECAY = 1
    DISCOUNT = 0.8
    DISCOUNT_INC = 0.995
    DISCOUNT_MAX = 0.8
    RMS_DECAY = 0.9
    RMS_EPSILON = 0.0001
    MOMENTUM = 0
    CLIP_NORM = 1.0
    EPSILON_START = 0
    EPSILON_MIN = 0
    EPSILON_DECAY = 100
    UPDATE_FREQUENCY = 3
    REPLAY_MEMORY_SIZE = 10000
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

    ENV = 'acrobot'

    # SCORE_FUNC = 'ranked_avg_knn_scores'
    SCORE_FUNC = 'avg_knn_scores'
    # KNN = 'batch_count_scaled_knn'
    KNN = 'batch_knn'

    # For loading network/dataset
    # START_COUNT = 500
    START_COUNT = 0
    # EXPERIMENT_DIR = os.path.join(os.getcwd(), 'experiments', 'pendulum novelty_reward_with_d_step_q_planning_2019-09-17 19-44-30_2905625')
    EXPERIMENT_DIR = None

    XTRA = ''
    # ----------------------
    # Representation NN parameters:
    # ----------------------

    # if LEARN_REPRESENTATION:
    DROPOUT_P = 0.1
    INTERNAL_DIM = 4
    BETA = 0.0
    CONSECUTIVE_DISTANCE = 0.5
    SLACK_RATIO = 8

    # if REWARD_TYPE == 'novelty_reward':
    K = 5

    # if ACTION_TYPE == 'd_step_q_planning':
    DEPTH = 5
    HIGHER_DIM_OBS = True

    ITERS_PER_UPDATE = 50000
    # ITERS_PER_UPDATE = 1

    # For plotting
    OFFLINE_PLOTTING = False

    # Priority replay
    EXP_PRIORITY = 0.0

    TRAIN_NSTEP = 1

    TRAIN_REWARD = False

    """
    Here we specify options for reward learning. Since we separate intrinsic and extrinsic
    rewards, we need to decide which part to learn for our Q value / Reward network.
    Options are:
    * only_secondary
    * only_primary
    * combined (primary for reward learning, combined for q learning)
    """
    REWARD_LEARNING = "combined"

    # Observations per state. DIFFERENT from timesteps per action.
    OBS_PER_STATE = 4
    MONITOR = False
    TRAIN_CSC_DIST = False

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # --- Parse parameters ---
    parameters = process_gym_args(sys.argv[1:], Defaults)

    parameters.env_name = f"{parameters.env} " + parameters.env_name

    max_steps = parameters.steps_per_epoch

    """
    We have three options for observations, reflected here and in input_dims for env:
    1. 6 dim full observation from environment
    2. high dim + timesteps_per_action > 1: 
        each action is performed timesteps_per_action number of times, and all those actions count as a single state
    3. high dim + obs_per_state > 1:
        each action is performed once, and we take obs_per_state - 1 states before current state and use the combination
        of the obs_per_state number of observations as state
    """
    timesteps_per_action = 4 if parameters.higher_dim_obs and parameters.obs_per_state <= 1 else 1

    parameters.replay_start_size = parameters.batch_size

    start_count = parameters.replay_start_size

    LOG_PERIODICITY = parameters.steps_per_epoch // 30

    # clearing up clutter for printing args
    if not parameters.learn_representation:
        del parameters.dropout_p
        del parameters.internal_dim
        del parameters.beta
        del parameters.consec_dist
        del parameters.slack_ratio

    if not parameters.reward_type == 'novelty_reward':
        del parameters.k

    if not parameters.action_type == 'd_step_q_planning':
        del parameters.depth

    seed = None
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

    job_id = parameters.job_id
    testing = job_id == str(0)

    continue_running = parameters.experiment_dir is not None and parameters.start_count > 0

    if continue_running:
        from nsrl.helper.data import Bunch

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
        parameters = new_parameters

        start_count = parameters.start_count

    # --- Create experiment directory
    h = parameters.env_name + '_' + parameters.job_id

    testing = job_id == str(0)
    # testing = True
    root_save_path = os.path.join(ROOT_DIR, "examples", "gym", "experiments")
    try:
        os.mkdir(root_save_path)
    except Exception:
        pass

    experiment_dir = os.path.join(root_save_path, h)

    try:
        os.mkdir(experiment_dir)
    except Exception:
        raise Exception("Experiment already exists")

    # Save parameters here
    param_dict = vars(copy.deepcopy(parameters))

    params_fname = os.path.join(experiment_dir, "parameters.json")
    with open(params_fname, 'w') as f:
        json.dump(param_dict, f)

    # --- Instantiate environment ---
    internal_dim = None if not hasattr(parameters, 'internal_dim') else parameters.internal_dim
    env = Control_env(rng,
                      env=parameters.env,
                      frame_skip=parameters.frame_skip, # Does nothing right now.
                      save_dir=experiment_dir,
                      intern_dim=internal_dim,
                      max_steps=max_steps,
                      monitor=parameters.monitor if hasattr(parameters, 'monitor') else False,
                      higher_dim_obs=parameters.higher_dim_obs,
                      obs_per_state=parameters.obs_per_state,
                      timesteps_per_action=timesteps_per_action,
                      seed=seed)
    dataset = None
    # reload dataset here if need be

    if hasattr(parameters, 'dataset_fname') and parameters.dataset_fname is not None and continue_running:
        from nsrl.helper.data import DataSet

        dataset = DataSet.load(parameters.dataset_fname)
        parameters.dataset_fname = None
        # this looks abysmal lmao
        # first env: Control_env, second env: StepMonitor, third env: TimeLimit
        wrapped_env = env.env.env
        wrapped_env._elapsed_steps = parameters.start_count
        dataset._environment.env.reinitialize(wrapped_env, experiment_dir, video_callable=lambda eid: True)
        env = dataset._environment

    score_func = ranked_avg_knn_scores
    if parameters.score_func == "avg_knn_scores":
        score_func = avg_knn_scores

    knn = batch_knn
    if parameters.knn == "batch_count_scaled_knn":
        knn = batch_count_scaled_knn

    parameters.score_func = score_func
    parameters.knn = knn

    learning_algo = NSRS(
        env,
        random_state=rng,
        high_dim_obs=parameters.higher_dim_obs,
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

    plotter = Plotter(experiment_dir,
                      env_name=h,
                      host=parameters.viz_host,
                      port=parameters.viz_port,
                      offline=parameters.offline_plotting)

    plotter.plot_text('hyperparams', stringify_params(parameters))

    train_q = parameters.action_type != 'd_step_reward_planning'
    agent = SEAgent(
        env,
        learning_algo,
        plotter,
        random_state=rng,
        train_policy=train_policy,
        test_policy=test_policy,
        train_q=train_q,
        train_rew=parameters.train_reward,
        secondary_rewards=True,
        gather_data=True,
        dataset=dataset,
        **vars(parameters))

    # --- Create unique filename for FindBestController ---
    h = hash(vars(parameters), hash_name="sha1")
    fname = "Acrobot_" + h
    print("The parameters hash is: {}".format(h))
    print("The parameters are: {}".format(parameters))

    checkpoint_freq = LOG_PERIODICITY

    # --- Bind controllers to the agent ---
    # Before every training epoch (periodicity=1), we want to print a summary of the agent's epsilon, discount and
    # learning rate as well as the training epoch number.
    agent.attach(bc.VerboseController(
        evaluate_on='action',
        periodicity=1))

    agent.attach(eh.ExtrinsicRewardPlottingController(
        plotter
    ))

    # Gather 1 epoch with of data
    # agent.runSingleEpisode(episodes=1)

    # parameters.replay_start_size = min(agent._dataset.n_elems, parameters.replay_start_size)

    checkpoint_start_count = parameters.start_count if continue_running else 0

    agent.gather_data = False

    agent.attach(bc.CheckpointController(
        evaluate_on='action',
        periodicity=checkpoint_freq,
        experiment_dir=experiment_dir,
        save_dataset=True,
        start_count=checkpoint_start_count
    ))
    # Currently only works with 1 timestep per action, n obs per action.
    measure_hash_counts = True
    if measure_hash_counts:
        agent.attach(eh.HashStateCounterController(
            plotter,
            evaluate_on='action',
            periodicity=1,
            input_dims=env.inputDimensions()[0]
        ))

    loss_plotting_sum_over = 1000
    periodicity = 10
    if testing:
        loss_plotting_sum_over = 10
        periodicity = 10

    agent.attach(eh.LossPlottingController(
        plotter,
        evaluate_on='train_step',
        periodicity=periodicity,
        sum_over=loss_plotting_sum_over
    ))
    if parameters.reward_type == 'novelty_reward':
        agent.attach(eh.NoveltyRewardController(
            evaluate_on='train_loop',
            periodicity=1,
            k=parameters.k,
            score_func=score_func,
            knn=knn,
            secondary=True
        ))
    elif parameters.reward_type == 'hash_count_reward':
        agent.attach(eh.HashCountRewardController(
            evaluate_on='action',
            periodicity=1,
            input_dims=env.inputDimensions()[0],
            secondary=True
        ))
    elif parameters.reward_type == 'transition_loss_reward':
        agent.attach(eh.TransitionLossRewardController(
            evaluate_on='train_loop',
            periodicity=1,
            secondary=True
        ))
    elif parameters.reward_type == 'rnd':
        agent.attach(eh.RNDRewardController(

        ))

    abstr_plotting_periodicity = checkpoint_freq
    abstr_plotting_evaluate = 'action'
    if testing:
        abstr_plotting_periodicity = 10000
        abstr_plotting_evaluate = 'train_step'

    if internal_dim is not None and internal_dim < 4:
        agent.attach(eh.AbstractRepPlottingController(
            plotter,
            evaluate_on=abstr_plotting_evaluate,
            start_count=start_count,
            periodicity=abstr_plotting_periodicity
        ))
    if hasattr(parameters, 'monitor') and parameters.monitor:
        agent.attach(bc.VideoRecordingController(
            periodicity=checkpoint_freq,
            evaluate_on='action',
            start_count=checkpoint_start_count
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

    agent.run(parameters.epochs, parameters.steps_per_epoch, break_on_done=True)

"""
ALE (more specifically, montezuma's revenge)

"""
import sys
import os
import logging
import json
import copy
import torch
import errno
import numpy as np
from joblib import hash, dump

from nsrl.default_parser import process_ale_args, stringify_params
from nsrl.agent import SEAgent
from nsrl.learning_algos.NSRS_pytorch import NSRS
from .ALE_env_gym import MyEnv as ALE_env
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
"""

class Defaults:
    # ----------------------
    # Experiment Parameters
    # ----------------------
    STEPS_PER_EPOCH = 25000
    EPOCHS = 1
    STEPS_PER_TEST = 25000
    PERIOD_BTW_SUMMARY_PERFS = 1

    # ----------------------
    # Environment Parameters
    # ----------------------
    FRAME_SKIP = 2

    # ----------------------
    # DQN Agent parameters:
    # ----------------------
    UPDATE_RULE = 'rmsprop'
    LEARNING_RATE = 0.0001
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
    UPDATE_FREQUENCY = 1
    REPLAY_MEMORY_SIZE = 25000
    BATCH_SIZE = 64
    FREEZE_INTERVAL = 5000
    DETERMINISTIC = True

    LEARN_REPRESENTATION = True

    # REWARD_TYPE = 'count_reward'
    REWARD_TYPE = 'novelty_reward'

    # ACTION_TYPE = '1_step_q_planning'
    ACTION_TYPE = 'd_step_q_planning'
    # ACTION_TYPE = 'd_step_reward_planning'
    # ACTION_TYPE = 'q_planning'
    # ACTION_TYPE = 'reward_argmax'
    # ACTION_TYPE = 'q_argmax'

    # SCORE_FUNC = 'ranked_avg_knn_scores'
    SCORE_FUNC = 'avg_knn_scores'
    # KNN = 'batch_count_scaled_knn'
    KNN = 'batch_knn'

    START_COUNT = 0
    EXPERIMENT_DIR = None

    XTRA = ''
    # ----------------------
    # Representation NN parameters:
    # ----------------------

    # if LEARN_REPRESENTATION:
    DROPOUT_P = 0.1
    INTERNAL_DIM = 3
    BETA = 0.0
    CONSECUTIVE_DISTANCE = 0.5
    SLACK_RATIO = 10

    # if REWARD_TYPE == 'novelty_reward':
    K = 5

    # if ACTION_TYPE == 'd_step_q_planning':
    DEPTH = 5

    HIGHER_DIM_OBS = True
    # ITERS_PER_UPDATE = 50000
    ITERS_PER_UPDATE = 30

    # For plotting
    OFFLINE_PLOTTING = False

    # Priority replay
    EXP_PRIORITY = 0.0

    # For loading network/dataset
    # NETWORK_FNAME = "model_simple maze novelty reward with d step reward planning_2019-06-20 11-01-32_1367306.epoch=300"
    # NETWORK_FNAME = os.path.join(os.getcwd(), 'nnets', 'to_save', NETWORK_FNAME)
    # DATASET_FNAME = "dataset_model_simple maze novelty reward with d step reward planning_2019-06-20 11-01-32_1367306.epoch=300.pkl"
    # DATASET_FNAME = os.path.join(os.getcwd(), 'nnets', 'to_save', DATASET_FNAME)
    #
    NETWORK_FNAME = None
    DATASET_FNAME = None

    USE_ITERATIVE_DATASET = False
    EPOCHS_PER_TRAIN = 25000

    TRAIN_NSTEP = 5

    GAME = "MontezumaRevenge-v0"

    REWARD_LEARNING = "combined"

    TRAIN_REWARD = True
    # Currently only supports either obs_per_state = 1 or timestep_per_action = 1
    OBS_PER_STATE = 1
    MONITOR = False
    ENV = 'montezumas_revenge'

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # --- Parse parameters ---
    parameters = process_ale_args(sys.argv[1:], Defaults)

    parameters.env_name = f"{parameters.env} " + parameters.env_name

    timesteps_per_action = 4 if parameters.higher_dim_obs and parameters.obs_per_state <= 1 else 1

    job_id = parameters.job_id
    testing = job_id == str(0)

    parameters.replay_start_size = parameters.batch_size #* 16  #if not testing else parameters.batch_size
    # parameters.replay_start_size = 1024

    start_count = parameters.replay_start_size

    LOG_PERIODICITY = parameters.steps_per_epoch // 500

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

    # --- Create experiment directory
    h = parameters.env_name + '_' + parameters.job_id

    root_save_path = os.path.join(ROOT_DIR, "experiments", "ALE", "runs")
    if not os.path.exists(root_save_path):
        os.mkdir(root_save_path)

    experiment_dir = os.path.join(root_save_path, h)

    try:
        os.makedirs(experiment_dir)
    except OSError as e:
        if e.errno == errno.EEXIST:
            raise OSError("Experiment already exists.")


    # Save parameters here
    param_dict = vars(copy.deepcopy(parameters))

    params_fname = os.path.join(experiment_dir, "parameters.json")
    with open(params_fname, 'w') as f:
        json.dump(param_dict, f)

    # --- Instantiate environment ---
    # internal_dim = None if not hasattr(parameters, 'internal_dim') else parameters.internal_dim
    env = ALE_env(rng, game=parameters.game,
                  frame_skip=parameters.frame_skip,
                  save_dir=experiment_dir,
                  intern_dim=parameters.internal_dim,
                  obs_per_state=parameters.obs_per_state,
                  timesteps_per_action=timesteps_per_action,
                  # crop=True, # THIS WAS CHANGED
                  reduced_actions=True, # THIS TOO
                  seed=seed)

    test_env = ALE_env(rng, game=parameters.game,
                      frame_skip=parameters.frame_skip,
                      save_dir=experiment_dir,
                      intern_dim=parameters.internal_dim,
                      obs_per_state=parameters.obs_per_state,
                      timesteps_per_action=timesteps_per_action,
                      # crop=True, # THIS WAS CHANGED
                      reduced_actions=True, # THIS TOO
                    )

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
        high_int_dim=True,
        train_csc_dist=True, # TESTING PURPOSES
        # train_linf_dist=True,
        scale_transition=True,
        transition_hidden_units=200,
        **vars(parameters))

    test_policy = EpsilonGreedyPolicy(learning_algo, env.nActions(), rng, parameters.epsilon_start)
    train_policy = EpsilonGreedyPolicy(learning_algo, env.nActions(), rng, parameters.epsilon_start)
    if parameters.action_type == 'q_argmax':
        test_policy = ep.QArgmaxPolicy(learning_algo, env.nActions(), rng, parameters.epsilon_start)
        train_policy = ep.QArgmaxPolicy(learning_algo, env.nActions(), rng, parameters.epsilon_start)
    elif parameters.action_type == 'd_step_q_planning':
        test_policy = ep.MCPolicy(learning_algo, env.nActions(), rng, depth=parameters.depth, epsilon_start=parameters.epsilon_start)
        train_policy = ep.MCPolicy(learning_algo, env.nActions(), rng, depth=parameters.depth, epsilon_start=0.0)
    elif parameters.action_type == 'bootstrap_q':
        test_policy = ep.BootstrapDQNPolicy(learning_algo, env.nActions(), rng, parameters.epsilon_start)
        train_policy = ep.BootstrapDQNPolicy(learning_algo, env.nActions(), rng, parameters.epsilon_start)

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
        train_rew=parameters.train_reward, # CHANGE THIS BACK
        secondary_rewards=True,
        gather_data=True,
        **vars(parameters))


    # --- Create unique filename for FindBestController ---
    h = hash(vars(parameters), hash_name="sha1")
    fname = "ALE_" + h
    print("The parameters hash is: {}".format(h))
    print("The parameters are: {}".format(parameters))

    checkpoint_freq = LOG_PERIODICITY

    # --- Bind controllers to the agent ---
    # Before every training epoch (periodicity=1), we want to print a summary of the agent's epsilon, discount and
    # learning rate as well as the training epoch number.
    agent.attach(bc.VerboseController(
        evaluate_on='action',
        periodicity=1))


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

    # For our intrinsic reward controllers we add them to our secondary rewards here.
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
            secondary=True,
            discrete=True
        ))
    elif parameters.reward_type == 'transition_loss_reward':
        agent.attach(eh.TransitionLossRewardController(
            evaluate_on='train_loop',
            periodicity=1,
            secondary=True
        ))

    # if hasattr(parameters, 'monitor') and parameters.monitor:
    #     agent.attach(bc.VideoRecordingController(
    #         periodicity=1000,
    #         evaluate_on='action',
    #     ))

    agent.attach(bc.TestRepresentationController(
        test_env,
        plotter,
        experiment_dir,
        evaluate_on='train_loop',
        episode_length=200,
        periodicity=1000,
        summarize_every=1
    ))

    # eval_every = checkpoint_freq
    eval_every = 100
    eval_on = 'action'
    viewing_history_size = 100
    agent.attach(bc.FramesToVideoController(
        experiment_dir,
        start_count=start_count if eval_on == 'train_loop' else 0,
        skip_first=start_count if eval_on == 'action' else 0,
        periodicity=eval_every,
        evaluate_on=eval_on,
        frames_per_video=viewing_history_size
    ))

    agent.attach(bc.CheckpointController(
        initial_count=parameters.replay_start_size,
        evaluate_on=eval_on,
        start_count=start_count if eval_on == 'train_loop' else 0,
        skip_first=start_count if eval_on == 'action' else 0,
        periodicity=eval_every,
        experiment_dir=experiment_dir,
        save_dataset=True,
        keep_every=10
    ))


    if parameters.internal_dim:
        abstr_rep_every = eval_every
        agent.attach(eh.AbstractRepPlottingController(
            plotter,
            evaluate_on=eval_on,
            start_count=start_count if eval_on == 'train_loop' else 0,
            skip_first=start_count if eval_on == 'action' else 0,
            periodicity=abstr_rep_every,
            limit_history=viewing_history_size
        ))

    # This NEEDS to go after the reward controllers for secondary reward.
    agent.attach(bc.RewardPlottingController(
        plotter,
        periodicity=1,
        evaluate_on='action',
        include_secondary_rewards=True
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


    agent.run(parameters.epochs, parameters.steps_per_epoch)


"""This module contains a function to help parse command-line arguments.

"""
import json
import argparse
from distutils import util
from datetime import datetime

def process_args(args, defaults, parser=None):
    """Handle the command line and return an object containing all the parameters.

    Arguments:
        args     - list of command line arguments (not including executable name)
        defaults - a name space with variables corresponding to each of the required default command line values.
    """
    if parser is None:
        parser = argparse.ArgumentParser()

    parser.add_argument('-e', '--epochs', dest="epochs", type=int,
                        default=defaults.EPOCHS,
                        help='Number of training epochs (default: %(default)s)')
    parser.add_argument('-s', '--steps-per-epoch', dest="steps_per_epoch",
                        type=int, default=defaults.STEPS_PER_EPOCH,
                        help='Number of steps per epoch (default: %(default)s)')
    parser.add_argument('-t', '--test-length', dest="steps_per_test",
                        type=int, default=defaults.STEPS_PER_TEST,
                        help='Number of steps per test (default: %(default)s)')
    parser.add_argument('-f', '--freq_summary_perfs', dest="period_btw_summary_perfs",
                        type=int, default=defaults.PERIOD_BTW_SUMMARY_PERFS,
                        help='freq summary perfs (default: %(default)s)')
    parser.add_argument('--frame-skip', dest="frame_skip",
                        default=defaults.FRAME_SKIP, type=int,
                        help='Every how many frames to process '
                        '(default: %(default)s)')
    parser.add_argument('--jobid', dest="job_id", type=str, default="0")
    parser.add_argument('--update-rule', dest="update_rule",
                        type=str, default=defaults.UPDATE_RULE,
                        help=('deepmind_rmsprop|rmsprop|sgd ' +
                              '(default: %(default)s)'))
    parser.add_argument('--learning-rate', dest="learning_rate",
                        type=float, default=defaults.LEARNING_RATE,
                        help='Learning rate (default: %(default)s)')
    parser.add_argument('--learning-rate-decay', dest="learning_rate_decay",
                        type=float, default=defaults.LEARNING_RATE_DECAY,
                        help='Learning rate (default: %(default)s)')
    parser.add_argument('--rms-decay', dest="rms_decay",
                        type=float, default=defaults.RMS_DECAY,
                        help='Decay rate for rms_prop (default: %(default)s)')
    parser.add_argument('--rms-epsilon', dest="rms_epsilon",
                        type=float, default=defaults.RMS_EPSILON,
                        help='Denominator epsilson for rms_prop ' +
                        '(default: %(default)s)')
    parser.add_argument('--momentum', type=float, default=defaults.MOMENTUM,
                        help=('Momentum term for Nesterov momentum. '+
                              '(default: %(default)s)'))
    parser.add_argument('--clip-norm', dest="clip_norm", type=float,
                        default=defaults.CLIP_NORM,
                        help=('Max L2 norm for the gradient. ' +
                              '(default: %(default)s)'))
    parser.add_argument('--discount', type=float, default=defaults.DISCOUNT,
                        help='Discount rate init')
    parser.add_argument('--discount_inc', type=float, default=defaults.DISCOUNT_INC,
                        help='Discount rate')
    parser.add_argument('--discount_max', type=float, default=defaults.DISCOUNT_MAX,
                        help='Discount rate max')
    parser.add_argument('--exp-priority', dest='exp_priority', type=float, default=defaults.EXP_PRIORITY)
    parser.add_argument('--epsilon-start', dest="epsilon_start",
                        type=float, default=defaults.EPSILON_START,
                        help=('Starting value for epsilon. ' +
                              '(default: %(default)s)'))
    parser.add_argument('--epsilon-min', dest="epsilon_min",
                        type=float, default=defaults.EPSILON_MIN,
                        help='Minimum epsilon. (default: %(default)s)')
    parser.add_argument('--epsilon-decay', dest="epsilon_decay",
                        type=float, default=defaults.EPSILON_DECAY,
                        help=('Number of steps to minimum epsilon. ' +
                              '(default: %(default)s)'))
    parser.add_argument('--max-history', dest="replay_memory_size",
                        type=int, default=defaults.REPLAY_MEMORY_SIZE,
                        help=('Maximum number of steps stored in replay ' +
                              'memory. (default: %(default)s)'))
    parser.add_argument('--batch-size', dest="batch_size",
                        type=int, default=defaults.BATCH_SIZE,
                        help='Batch size. (default: %(default)s)')
    parser.add_argument('--freeze-interval', dest="freeze_interval",
                        type=int, default=defaults.FREEZE_INTERVAL,
                        help=('Interval between target freezes. ' +
                              '(default: %(default)s)'))
    parser.add_argument('--update-frequency', dest="update_frequency",
                        type=int, default=defaults.UPDATE_FREQUENCY, help=('Number of actions before each SGD update. '+ '(default: %(default)s)'))
    parser.add_argument('--deterministic', dest='deterministic', action='store_true',
                        help=('If fixed seed (default: %(default)s)'))
    parser.add_argument('--no-deterministic', dest='deterministic', action='store_false',
                        help=('If no fixed seed'))

    parser.set_defaults(deterministic=defaults.DETERMINISTIC)
    parser.add_argument('--param1', dest="param1") # Additional parameter depending on the environment
    parser.add_argument('--param2', dest="param2") # Additional parameter depending on the environment
    parser.add_argument('--param3', dest="param3") # Additional parameter depending on the environment

    parameters = parser.parse_args(args)

    return parameters

def process_se_args(args, defaults):

    parser = argparse.ArgumentParser()

    try:
        parser.add_argument('--k', dest='k', type=int, default=defaults.K)
    except AttributeError:
        pass

    try:
        parser.add_argument('--beta', dest='beta', type=float, default=defaults.BETA)
        parser.add_argument('--internal-dim', dest='internal_dim', type=int, default=defaults.INTERNAL_DIM)
        parser.add_argument('--dropout-p', dest='dropout_p', type=float, default=defaults.DROPOUT_P)
        parser.add_argument('--consecutive-distance', dest='consec_dist', type=float, default=defaults.CONSECUTIVE_DISTANCE)
        parser.add_argument('--slack-ratio', dest='slack_ratio', type=float, default=defaults.SLACK_RATIO)
        parser.add_argument('--encoder-prop-td', dest='encoder_prop_td',
                            type=lambda x:bool(util.strtobool(x)), default=defaults.ENCODER_PROP_TD)
        parser.add_argument('--rnn-q-func', dest='rnn_q_func',
                            type=lambda x:bool(util.strtobool(x)), default=defaults.RNN_Q_FUNC)
    except AttributeError:
        pass

    try:
        parser.add_argument('--depth', dest='depth', type=int, default=defaults.DEPTH)
    except AttributeError:
        pass

    # parser.add_argument('--use-iterative-dataset', dest='use_iterative_dataset', type=lambda x:bool(util.strtobool(x)),
    #                     default=defaults.USE_ITERATIVE_DATASET)
    # parser.add_argument('--epochs-per-train', dest='epochs_per_train', type=int,
    #                     default=defaults.EPOCHS_PER_TRAIN)

    parser.add_argument('--train-nstep', dest='train_nstep', type=int, default=defaults.TRAIN_NSTEP)

    parser.add_argument('--size-maze', dest='size_maze', type=int, default=defaults.SIZE_MAZE)
    parser.add_argument('--maze-walls', dest='maze_walls',
                        type=lambda x:bool(util.strtobool(x)), default=defaults.MAZE_WALLS if hasattr(defaults, 'MAZE_WALLS') else False)

    parser.add_argument('--learn-representation', dest='learn_representation',
                        type=lambda x:bool(util.strtobool(x)), default=defaults.LEARN_REPRESENTATION)
    parser.add_argument('--reward-type', dest='reward_type', type=str, default=defaults.REWARD_TYPE)
    parser.add_argument('--score-func', dest='score_func', type=str, default=defaults.SCORE_FUNC)
    parser.add_argument('--knn', dest='knn', type=str, default=defaults.KNN)
    parser.add_argument('--action-type', dest='action_type', type=str, default=defaults.ACTION_TYPE)

    parser.add_argument('--iters-per-update', dest='iters_per_update', type=int, default=defaults.ITERS_PER_UPDATE)
    parser.add_argument('--higher-dim-obs', dest='higher_dim_obs', type=lambda x:bool(util.strtobool(x)),
                        default=defaults.HIGHER_DIM_OBS)
    parser.add_argument('--description', dest='extra_description', type=str, default=defaults.XTRA)

    parser.add_argument('--viz-host', dest='viz_host', default='localhost')
    parser.add_argument('--viz-port', dest='viz_port', default=8097)
    parser.add_argument('--offline-plotting', dest='offline_plotting', type=lambda x:bool(util.strtobool(x)),
                        default=defaults.OFFLINE_PLOTTING)

    parser.add_argument('--experiment-dir', dest='experiment_dir', type=str, default=defaults.EXPERIMENT_DIR)
    parser.add_argument('--start-count', dest='start_count', type=int, default=defaults.START_COUNT)

    parameters = process_args(args, defaults, parser=parser)

    date_time = datetime.now().strftime("%Y-%m-%d %H-%M-%S")
    if parameters.reward_type is not None:
        parameters.env_name = parameters.reward_type.replace('_', ' ') + ' with ' + parameters.action_type.replace('_', ' ') + '_' + date_time
    else:
        parameters.env_name = parameters.action_type.replace('_', ' ') + '_' + date_time

    return parameters

def process_gym_args(args, defaults, parser=argparse.ArgumentParser()):
    try:
        parser.add_argument('--k', dest='k', type=int, default=defaults.K)
    except AttributeError:
        pass

    try:
        parser.add_argument('--train-reward', dest='train_reward',
                            type=lambda x: bool(util.strtobool(x)), default=defaults.TRAIN_REWARD)
        parser.add_argument('--beta', dest='beta', type=float, default=defaults.BETA)
        parser.add_argument('--internal-dim', dest='internal_dim', type=int, default=defaults.INTERNAL_DIM)
        parser.add_argument('--dropout-p', dest='dropout_p', type=float, default=defaults.DROPOUT_P)
        parser.add_argument('--consecutive-distance', dest='consec_dist', type=float,
                            default=defaults.CONSECUTIVE_DISTANCE)
        parser.add_argument('--slack-ratio', dest='slack_ratio', type=float, default=defaults.SLACK_RATIO)
    except AttributeError:
        pass

    try:
        parser.add_argument('--depth', dest='depth', type=int, default=defaults.DEPTH)
    except AttributeError:
        pass
    parser.add_argument('--monitor', dest='monitor',
                        type=lambda x: bool(util.strtobool(x)), default=defaults.MONITOR)
    parser.add_argument('--env', dest='env', type=str, default=defaults.ENV)
    parser.add_argument('--obs-per-state', dest='obs_per_state', type=int, default=defaults.OBS_PER_STATE)
    parser.add_argument('--reward-learning', dest='reward_learning', type=str, default=defaults.REWARD_LEARNING)

    parser.add_argument('--train-nstep', dest='train_nstep', type=int, default=defaults.TRAIN_NSTEP)

    parser.add_argument('--learn-representation', dest='learn_representation',
                        type=lambda x: bool(util.strtobool(x)), default=defaults.LEARN_REPRESENTATION)
    parser.add_argument('--reward-type', dest='reward_type', type=str, default=defaults.REWARD_TYPE)
    parser.add_argument('--action-type', dest='action_type', type=str, default=defaults.ACTION_TYPE)

    parser.add_argument('--knn', dest='knn', type=str, default=defaults.KNN)
    parser.add_argument('--score-func', dest='score_func', type=str, default=defaults.SCORE_FUNC)

    parser.add_argument('--iters-per-update', dest='iters_per_update', type=int, default=defaults.ITERS_PER_UPDATE)
    parser.add_argument('--higher-dim-obs', dest='higher_dim_obs', type=lambda x: bool(util.strtobool(x)),
                        default=defaults.HIGHER_DIM_OBS)
    parser.add_argument('--description', dest='extra-description', type=str, default=defaults.XTRA)

    parser.add_argument('--experiment-dir', dest='experiment_dir', type=str, default=defaults.EXPERIMENT_DIR)
    parser.add_argument('--start-count', dest='start_count', type=int, default=defaults.START_COUNT)
    parser.add_argument('--viz-host', dest='viz_host', default='localhost')
    parser.add_argument('--viz-port', dest='viz_port', default=8097)
    parser.add_argument('--offline-plotting', dest='offline_plotting', type=lambda x: bool(util.strtobool(x)),
                        default=defaults.OFFLINE_PLOTTING)

    parameters = process_args(args, defaults, parser=parser)

    date_time = datetime.now().strftime("%Y-%m-%d %H-%M-%S")

    parameters.env_name = parameters.reward_type + '_with_' + parameters.action_type + '_' + date_time

    return parameters


def process_ale_args(args, defaults):
    parser = argparse.ArgumentParser()

    parser.add_argument('--game', dest='game', type=str, default=defaults.GAME)

    parameters = process_gym_args(args, defaults, parser=parser)

    date_time = datetime.now().strftime("%Y-%m-%d %H-%M-%S")

    parameters.env_name = parameters.reward_type + '_with_' + parameters.action_type + '_' + date_time

    return parameters


def get_args_dict(args):
    builtin = ('basestring', 'bool', 'complex', 'dict', 'float', 'int',
               'list', 'long', 'str', 'tuple')
    args_dict = {k: v for k, v in vars(args).items()
                 if type(v).__name__ in builtin}
    return args_dict


def stringify_params(namespace):
    d = get_args_dict(namespace)
    text = json.dumps(d)
    text = text.replace(',', '<br>')
    return text

if __name__ == '__main__':
    pass

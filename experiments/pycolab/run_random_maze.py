import os
import json

import numpy as np
from random import randrange
from .maze_env import MazeEnv
from definitions import ROOT_DIR

import matplotlib.pyplot as plt

if __name__ == "__main__":
    size_maze = 15
    steps_per_episode = 5000

    measure_exploration = size_maze != 15

    episodes = 10

    env = MazeEnv(size_maze=size_maze)
    env.reset()
    steps_to_completion = []
    exploration_factor = np.array([])
    ratio_states_visited = np.array([])

    if measure_exploration:
        empty_tiles = env._observation.layers[' ']
        num_empty_tiles = np.sum(empty_tiles.astype(int))

        exploration_factor = np.zeros((episodes, steps_per_episode + 1))
        exploration_factor[:, 0] = 1
        ratio_states_visited = np.zeros((episodes, steps_per_episode + 1))
        ratio_states_visited[:, 0] = 1 / num_empty_tiles

    for ep in range(episodes):
        for i in range(steps_per_episode):
            action = randrange(env.nActions())
            env.act(action)
            trajectory = env.trajectory

            if measure_exploration:
                num_unique_states = np.unique(trajectory).shape[0]

                exploration_factor[ep, i + 1] = num_unique_states / len(trajectory)
                ratio_states_visited[ep, i + 1] = num_unique_states / num_empty_tiles

            if env.inTerminalState():
                steps_to_completion.append(i)
                break

        env.reset()
    plots_dir = os.path.join(ROOT_DIR, "experiments", 'pycolab', 'plots')
    if measure_exploration:
        if not os.path.isdir(plots_dir):
            os.mkdir(plots_dir)

    baselines_dir = os.path.join(plots_dir, 'baselines')
    if not os.path.isdir(baselines_dir):
        os.mkdir(baselines_dir)

    fname = os.path.join(baselines_dir, f'random_agent_{size_maze}.json')

    data_to_save = {
        "exploration_factors": exploration_factor.tolist(),
        "ratios_visited": ratio_states_visited.tolist(),
        "steps_to_completion": steps_to_completion
    }

    with open(fname, 'w') as f:
        json.dump(data_to_save, f)

    if measure_exploration:
        x = np.arange(0, steps_per_episode + 1)
        avg_exp_factor = np.average(exploration_factor, axis=0)
        avg_ratio_states_visited = np.average(ratio_states_visited, axis=0)

        ax1 = plt.subplot(1, 2, 1)
        ax1.plot(x, avg_exp_factor)
        ax1.set_ylim([0, 1])

        ax2 = plt.subplot(1, 2, 2)
        ax2.plot(x, avg_ratio_states_visited)
        ax2.set_ylim([0, 1])
        plt.show()


    print("here")


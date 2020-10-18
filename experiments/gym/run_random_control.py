import os
import errno
import numpy as np
from datetime import datetime
from random import randrange
from .control_env import MyEnv as Control_env
from definitions import ROOT_DIR


if __name__ == "__main__":
    episodes = 2
    steps_per_episode = 10000
    rng = np.random.RandomState()
    env = 'acrobot'

    root_save_path = os.path.join(ROOT_DIR, "experiments", "gym", "runs")

    date_time = datetime.now().strftime("%Y-%m-%d %H-%M-%S")
    experiment_dir = os.path.join(root_save_path, f'{env} random_eps_{episodes}_steps_{steps_per_episode}_{date_time}')

    try:
        os.makedirs(experiment_dir)
    except OSError as e:
        if e.errno == errno.EEXIST:
            raise OSError("Experiment already exists.")


    env = Control_env(rng, max_steps=steps_per_episode,
                      higher_dim_obs=False, timesteps_per_action=1,
                      monitor=True, save_dir=experiment_dir, env=env)
    steps_to_completion = []
    recording_num = 0
    record_every = 500

    # to_load = [-0.02917805, 2.44747804, 0.46010804, 0.03353835]
    for t in range(episodes):
        done = False
        env.reset()
        # env.env.unwrapped.state = np.array(to_load)
        s = 0
        while not done:
            action = randrange(env.nActions())
            r = env.act(action)
            done = env.inTerminalState()
            # if s > 0 and s % 500 == 0 and r < 0:
            #     print("resetting video recorder")
            #     recording_num += 1
            #     env.env.reset_video_recorder()
            s += 1
        steps_to_completion.append(s)

    print(f"average of {episodes} is {np.average(steps_to_completion)}")
    print("done")


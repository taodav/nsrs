import numpy as np
import copy
import gym
import torch
import plotly.graph_objs as go
import plotly.io as pio
import cv2

from nsrl.base_classes import Environment
from nsrl.helper.pytorch import device
from nsrl.helper.gym_env import StepMonitor, PickleableEnv

import matplotlib.pyplot as plt

class MyEnv(Environment):
    def __init__(self, rng, save_dir='default', monitor=True, intern_dim=2, higher_dim_obs=False,
                 timesteps_per_action=1, obs_per_state=1, env='acrobot', seed=None, **kwargs):
        """ Initialize environment.

        Arguments:
            rng - the numpy random number generator
        """
        id = 'AcrobotModified-v1'
        entry_point = 'nsrl.helper.gym_env:ContinuableAcrobotEnv'

        if env == 'pendulum':
            id = 'PendulumModified-v1'
            entry_point = 'nsrl.helper.gym_env:ContinuablePendulumEnv'
        max_steps = kwargs.get('max_steps', 200)
        gym.envs.register(
            id=id,
            entry_point=entry_point,
            max_episode_steps=max_steps,
        )

        self.env = gym.make(id)
        if seed is not None:
            self.env.seed(seed)
        self._discrete_actions = hasattr(self.env.action_space, 'n')
        self._mapping = None

        # Currently default to discretizing action space to 4 actions
        self._n_discrete_actions = 4
        if monitor:
            self.env = StepMonitor(self.env, save_dir, video_callable=lambda eid: True, env_name=env)
        else:
            self.env = PickleableEnv(self.env)

        if not self._discrete_actions:
            high = self.env.action_space.high[0]
            low = self.env.action_space.low[0]
            a_unit = (high - low) / (self._n_discrete_actions - 1)
            self._mapping = {i: [low + i * a_unit] for i in range(self._n_discrete_actions)}

        self._frame_size = (32, 32)
        # we save the experiment directory here for reloading purposes (see reloading dataset in agent)
        self.save_dir = save_dir
        self.rng = rng
        self._last_observation = None
        self.is_terminal = False
        self._higher_dim_obs = higher_dim_obs
        # self._input_dim = [(obs_per_state,) + self.env.observation_space.shape]  # self.env.observation_space.shape is equal to 2
        if self._higher_dim_obs:
            size = self._frame_size
            if timesteps_per_action > 1:
                size = (1, timesteps_per_action, ) + size
            elif obs_per_state >= 1:
                size = (obs_per_state, ) + size
            self._input_dim = [size]
        self._intern_dim = intern_dim
        self._save_dir = save_dir

        self._screen, self._reduced_screen = None, None

        # and we use only the current observation in the pseudo-state

        self._timesteps_per_action = kwargs.get('timesteps_per_action', 1)

    def act(self, action):
        """ Simulate one time step in the environment.
        """
        reward = 0
        self.state = np.zeros((self._timesteps_per_action, self._frame_size[0], self._frame_size[1]), dtype=np.float)
        for t in range(self._timesteps_per_action):
            if self._mapping is not None:
                action = self._mapping[action]
            self._last_observation, r, self.is_terminal, info = self.env.step(action)
            reward += r

            if (self.mode == 0):  # Show the policy only at test time
                try:
                    self.env.render()
                except:
                    pass
                    # print("Warning:", sys.exc_info()[0])

            if self._higher_dim_obs:
                self._screen=np.average(self.env.render(mode='rgb_array'),axis=-1)
                self._reduced_screen = cv2.resize(self._screen, self._frame_size, interpolation=cv2.INTER_LINEAR)
                if self._timesteps_per_action > 1:
                    self.state[t, :, :] = self._reduced_screen
                else:
                    self.state = self._reduced_screen

        return reward / self._timesteps_per_action

    def reset(self, mode=0):
        """ Reset environment for a new episode.

        Arguments:
        Mode : int
            -1 corresponds to training and 0 to test
        """
        self.mode = mode

        self._last_observation = self.env.reset()
        if self._higher_dim_obs:
            rendering = self.env.render(mode='rgb_array')
            self._screen = np.average(rendering, axis=-1)
            self._reduced_screen = cv2.resize(self._screen, self._frame_size, interpolation=cv2.INTER_LINEAR)
            initState = copy.deepcopy(self._reduced_screen)
            if self._timesteps_per_action > 1:
                self.state = np.repeat(initState[None, :, :], self._timesteps_per_action, axis=0)
            else:
                self.state = initState

        self.is_terminal = False

        return self._last_observation

    def inTerminalState(self):
        """ Tell whether the environment reached a terminal state after the last transition (i.e. the last transition
        that occured was terminal).
        """
        return self.is_terminal

    def plot_current_state(self):
        state = self.env.render(mode='rgb_array')
        self.plot_state(state)

    def plot_state(self, state):
        plt.imshow(state, cmap='gray')
        plt.show()

    def inputDimensions(self):
        return copy.deepcopy(self._input_dim)

    def nActions(self):
        if not self._discrete_actions:
            return self._n_discrete_actions

        return self.env.action_space.n

    def observe(self):
        if self._higher_dim_obs:
            return [(np.array(self.state) - 128) / 128]
        return [copy.deepcopy(self._last_observation)]

    def summarizePerformance(self, test_data_set, learning_algo, *args, **kwargs):
        """
        Summarize performance uses plotly. This call requires a learnt representation.
        :param test_data_set:
        :param learning_algo:
        :return:
        """

        save_image = kwargs.get('save_image', False)
        action_meanings = ['+1', '0', '-1']

        with torch.no_grad():
            for m in learning_algo.all_models: m.eval()

            test_observations = test_data_set.observations()[0] # b x intern_dim
            test_observations = torch.from_numpy(test_observations).float().to(device)
            test_abs_states = learning_algo.encoder.predict(test_observations)
            np_test_abs_states = test_abs_states.detach().cpu().numpy()

            x = np_test_abs_states[:, 0]
            y = np_test_abs_states[:, 1]
            z = np.zeros_like(y)
            if (self._intern_dim == 3):
                z = np_test_abs_states[:, 2]
            print("summarizing performance")
            trans_by_action_idx = []
            stacked_transitions = np.eye(self.nActions())
            # each element of this list should be a transition

            for one_hot_action in stacked_transitions:
                repeated_one_hot_actions = torch.from_numpy(np.repeat(one_hot_action[None, :], test_abs_states.shape[0], axis=0)).float().to(device)
                res = torch.cat([test_abs_states, repeated_one_hot_actions], dim=-1)
                transitions = learning_algo.transition(res).detach().cpu().numpy()

                trans_by_action_idx.append(transitions)

        trace_data = []
        opacity_unit = 1 / self.nActions()
        opacities = [(i + 1) * opacity_unit for i in range(self.nActions())]
        if self._intern_dim == 2:
            for trans, aname, opacity in zip(trans_by_action_idx, action_meanings, opacities):
                plot_x = []
                plot_y = []
                for x_o, y_o, x_y_n in zip(x, y, trans):
                    plot_x += [x_o, x_y_n[0], None]
                    plot_y += [y_o, x_y_n[1], None]
                trace_data.append(
                    go.Scatter(x=plot_x,
                               y=plot_y,
                               line=dict(color='rgba(0, 0, 0, ' + str(opacity) + ')'),
                               marker=dict(size=1),
                               name=aname))
            unit = 256 // len(x)
            scatter = go.Scatter(x=x, y=y, mode='markers+text',
                                 marker=dict(symbol='x', size=10,
                                             color=[f"rgb({int(i * unit)}, {int(unit * (len(x) - i))}, 0)" for i in range(len(x))]),
                                 text=list(range(len(x))),
                                 textposition='top center')
            trace_data.append(scatter)

        elif self._intern_dim == 3:
            for trans, aname, opacity in zip(trans_by_action_idx, action_meanings, opacities):
                plot_x = []
                plot_y = []
                plot_z = []
                for x_o, y_o, z_o, x_y_z_n in zip(x, y, z, trans):
                    plot_x += [x_o, x_y_z_n[0], None]
                    plot_y += [y_o, x_y_z_n[1], None]
                    plot_z += [z_o, x_y_z_n[2], None]

                trace_data.append(
                    go.Scatter3d(
                        x=plot_x, y=plot_y, z=plot_z,
                        line=dict(color='rgba(0, 0, 0, ' + str(opacity) + ')'),
                        marker=dict(size=1),
                        name=aname))
            unit = 256 // len(x)
            scatter = go.Scatter3d(x=x, y=y, z=z,
                                   mode='markers+text',
                                   text=list(range(len(x))),
                                   textposition='top center',
                                   marker=dict(symbol='circle',
                                               size=3,
                                               color=[f"rgb({int(i * unit)}, {int(unit * (len(x) - i))}, 0)" for i in range(len(x))]))
            trace_data.append(scatter)
        fig = dict(data=trace_data)

        if save_image:
            pio.write_image(fig, 'pytorch/fig_base_'+str(learning_algo.repr_update_counter)+'.png')

        return fig



def main():
    # This function can be used for debug purposes
    rng = np.random.RandomState(123456)
    myenv = MyEnv(rng)

    print(myenv.observe())


if __name__ == "__main__":
    main()

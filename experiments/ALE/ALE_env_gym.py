""" Interface with the ALE environment

"""
import numpy as np

import cv2
import copy
import torch
import gym
import plotly.graph_objects as go
import plotly.io as pio

from nsrl.base_classes import Environment
from nsrl.helper.pytorch import device, calculate_large_batch
import matplotlib.pyplot as plt
from nsrl.helper.gym_env import StepMonitor, PickleableEnv
from nsrl.helper.dim import pca

class MyEnv(Environment):
    VALIDATION_MODE = 0

    def __init__(self, rng, monitor=True, timesteps_per_action=1, obs_per_state=1, save_dir='default',
                 higher_dim_obs=True, seed=None, **kwargs):
        """ ALE gym environment.

        Arguments:
            rng - the numpy random number generator            
        """
        if(bool(kwargs["game"])):
            self.env = gym.make(kwargs["game"])
        else:
            # Choice between Seaquest-v4, Breakout-v4, SpaceInvaders-v4, BeamRider-v4, Qbert-v4, Freeway-v4', etc.
            self.env = gym.make('Seaquest-v4')

        if seed is not None:
            self.env.seed(seed)

        if monitor:
            self.env = StepMonitor(self.env, save_dir, video_callable=lambda eid: True, pickleable=False)
        else:
            self.env = PickleableEnv(self.env)

        max_steps = kwargs.get('max_steps', None)
        if max_steps is not None:
            self.setMaxSteps(max_steps)

        self._monitor = monitor
        self._random_state=rng
        frame_skip=kwargs.get('frame_skip',1)
        self._frame_skip = frame_skip if frame_skip >= 1 else 1


        self._frame_size = kwargs.get('frame_size', (64, 64))
        self._use_resnet = kwargs.get('use_resnet', False)

        if self._use_resnet:
            from torchvision.transforms import ToPILImage, Normalize, Resize, Compose
            from PIL import Image
            self._transforms = Compose([ToPILImage(),
                                        Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
                                       Resize((128, 128),
                                              interpolation=Image.NEAREST)])
            self._frame_size = (128, 128)

        self._higher_dim_obs = higher_dim_obs
        self._crop = kwargs.get('crop', False)
        self._action_mapping = []
        self._reduced_actions = kwargs.get('reduced_actions', False)
        if self._reduced_actions:
            self._action_mapping = [0, 1, 2, 3, 4, 5, 11, 12]

        # self._input_dim = [(obs_per_state,) + self.env.observation_space.shape]  # self.env.observation_space.shape is equal to 2
        if self._higher_dim_obs:
            size = self._frame_size
            if timesteps_per_action > 1:
                size = (1, timesteps_per_action, ) + size
            elif obs_per_state >= 1:
                size = (obs_per_state, ) + size
            self._input_dim = [size]
        self._screen=np.average(self.env.render(mode='rgb_array'),axis=-1)
        self._reduced_screen = cv2.resize(self._screen, self._frame_size, interpolation=cv2.INTER_LINEAR)
        self._intern_dim = kwargs.get('intern_dim', 2)
        self._timesteps_per_action = timesteps_per_action

        self._mode = -1
        self._mode_score = 0.0
        self._mode_episode_count = 0

    def setMaxSteps(self, steps):
        steps = self._timesteps_per_action * steps
        if self._monitor:
            self.env.env._max_episode_steps = steps
        else:
            self.env._max_episode_steps = steps

    @staticmethod
    def crop_center(image):
        h, w = image.shape
        if h == w:
            return image

        diff = h - w
        side_diff = diff // 2

        if diff > 0:
            return image[side_diff:h - side_diff]
        elif diff < 0:
            return image[:, -side_diff:w + side_diff]

    @staticmethod
    def crop_bottom(image):
        w = min(image.shape)
        return image[:w, :w]

    def reset(self, mode):
        if mode == self._mode:
            # already in the right mode
            self._mode_episode_count += 1
        else:
            # switching mode
            self._mode = mode
            self._mode_score = 0.0
            self._mode_episode_count = 0

        self._last_observation = self.env.reset()
        if self._higher_dim_obs:
            rendering = self.env.render(mode='rgb_array')
            self._screen = np.average(rendering, axis=-1)
            if self._crop:
                self._screen = self.crop_bottom(self._screen)
            self._reduced_screen = cv2.resize(self._screen, self._frame_size, interpolation=cv2.INTER_LINEAR)
            initState = copy.deepcopy(self._reduced_screen)
            if self._timesteps_per_action > 1:
                self.state = np.repeat(initState[None, :, :], self._timesteps_per_action, axis=0)
            else:
                self.state = initState

        self.terminal = False

        return self._last_observation

        
    def act(self, action):
        #print "action"
        #print action
        state_dim = self._frame_size
        if self._timesteps_per_action > 1:
            state_dim = (self._timesteps_per_action, ) + self._frame_size
        self.state = np.zeros(state_dim, dtype=np.float)
        reward=0

        if self._reduced_actions:
            action = self._action_mapping[action]

        for t in range(self._timesteps_per_action):
            observation, r, self.terminal, info = self.env.step(action)
            reward+=r
            if self.inTerminalState():
                break

            self._screen = np.average(observation,axis=-1) # Gray levels
            if self._crop:
                self._screen = self.crop_bottom(self._screen)
            self._reduced_screen = cv2.resize(self._screen, self._frame_size, interpolation=cv2.INTER_NEAREST)
            if self._timesteps_per_action > 1:
                self.state[t, :, :] = self._reduced_screen
            else:
                self.state = self._reduced_screen

        self._mode_score += reward

        # Set preset actions herud
        return reward

    def _calc_abstr_states_and_transitions(self, observations, learning_algo, dim_reduction_fn=None):
        with torch.no_grad():
            for m in learning_algo.all_models: m.eval()

            test_abs_states = calculate_large_batch(learning_algo.encoder, observations)
            np_test_abs_states = test_abs_states.detach().cpu().numpy()

            # We use PCA dimensionality reduction if our internal dimensions are too high
            if dim_reduction_fn is not None:
                np_test_abs_states = dim_reduction_fn(np_test_abs_states, 3)

            x = np_test_abs_states[:, 0]
            y = np_test_abs_states[:, 1]
            z = np.zeros_like(y)
            if (self._intern_dim == 3):
                z = np_test_abs_states[:, 2]

            print("== Logging abstract representation ==")
            trans_by_action_idx = []
            stacked_transitions = np.eye(self.nActions())
            # each element of this list should be a transition

            for one_hot_action in stacked_transitions:
                repeated_one_hot_actions = torch.from_numpy(
                    np.repeat(one_hot_action[None, :], test_abs_states.shape[0], axis=0)).float().to(device)
                res = torch.cat([test_abs_states, repeated_one_hot_actions], dim=-1)
                transitions = calculate_large_batch(learning_algo.transition, res).detach().cpu().numpy()
                if dim_reduction_fn is not None:
                    transitions = dim_reduction_fn(transitions, 3)

                trans_by_action_idx.append(transitions)

        return (x, y, z), trans_by_action_idx

    def plot_reprs(self, x, y, z, trans_by_action_idx, labels,
                   action_meanings, internal_dim=2):

        trace_data = []
        opacity_unit = 1 / self.nActions()
        opacities = [(i + 1) * opacity_unit for i in range(self.nActions())]

        if internal_dim == 2:
            if trans_by_action_idx:
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
                                 text=labels,
                                 textposition='top center',
                                 marker=dict(symbol='x', size=10,
                                             color=[f"rgb({int(i * unit)}, {int(unit * (len(x) - i))}, 0)" for i in
                                                    range(len(x))]))
            trace_data.append(scatter)

        elif internal_dim >= 3:
            if trans_by_action_idx:
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
                                   text=labels,
                                   textposition='top center',
                                   marker=dict(symbol='circle',
                                               size=3,
                                               color=[f"rgb({int(i * unit)}, {int(unit * (len(x) - i))}, 0)" for i in
                                                      range(len(x))]))
            trace_data.append(scatter)
        fig = dict(data=trace_data)
        return fig

    def summarizePerformance(self, test_data_set, learning_algo, *args, **kwargs):
        """
        Summarize performance uses plotly. This call requires a learnt representation.
        :param test_data_set:
        :param learning_algo:
        :param args:
        :param kwargs:
        :return:
        """
        save_image = kwargs.get('save_image', False)
        n_observations = kwargs.get('n_observations', -1)
        all_action_meanings = self.env.unwrapped.get_action_meanings()
        action_meanings = all_action_meanings[:self.nActions()]
        if self._reduced_actions:
            action_meanings = [all_action_meanings[a] for a in self._action_mapping]

        if self.inTerminalState() == False:
            self._mode_episode_count += 1

        test_observations = test_data_set.observationsMatchingBatchDim()[0] # b x intern_dim
        if n_observations > 0:
            test_observations = test_observations[-n_observations:]

        points, transitions = self._calc_abstr_states_and_transitions(test_observations, learning_algo)
        x, y, z = points

        labels = list(range(len(test_observations)))
        if n_observations > 0:
            labels = list(range(test_data_set.n_elems - n_observations, test_data_set.n_elems))

        fig = self.plot_reprs(x, y, z, transitions, labels, action_meanings, internal_dim=self._intern_dim)

        if save_image:
            pio.write_image(fig, 'pytorch/fig_base_'+str(learning_algo.repr_update_counter)+'.png')

        return fig

    def inputDimensions(self):
        return copy.deepcopy(self._input_dim)

    def observationType(self, subject):
        return np.uint8

    def nActions(self):
        """
        Actions:
        0: noop, 1: fire (jump), 2: up,
        3: right, 4: left, 5: down
        :return:
        """
        if self._reduced_actions:
            return len(self._action_mapping)
        # return 6
        return self.env.action_space.n

    def observe(self):
        return [np.array(self.state)]

    def inTerminalState(self):
        return self.terminal

    def plot_current_state(self):
        self.plot_state(self._reduced_screen)

    @classmethod
    def plot_state(cls, state):
        """
        plots the last frame in a "game state"
        :param state: frames_per_state x frame_size*
        """
        assert len(state.shape) == 3
        cls.plot_observation(state[-1])

    @staticmethod
    def plot_observation(observation):
        plt.imshow(observation, cmap='gray')
        plt.show()


if __name__ == "__main__":
    pass
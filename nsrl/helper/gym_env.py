import gym
import os
import numpy as np
from gym.wrappers import Monitor
from gym import Wrapper
from gym.envs.classic_control import AcrobotEnv, PendulumEnv


class PickleableEnv(Wrapper):
    def __init__(self, env, env_name='acrobot', **kwargs):
        super(PickleableEnv, self).__init__(env)
        self._env_name = env_name

        self._prev_saved_state = None

    def __getstate__(self):

        state = {
            'env_state': self.unwrapped.state,
            'env_np_random': self.unwrapped.np_random,
            'reset_state': True
        }
        if hasattr(self, 'last_u'):
            state['last_u'] = self.last_u

        return state

    def __setstate__(self, state):
        self._reset_count = state['reset_count']
        self._prev_saved_state = state

    def reinitialize(self, env, directory, video_callable=None):
        assert self._prev_saved_state is not None
        super(PickleableEnv, self).__init__(env)

        self.unwrapped.np_random = self._prev_saved_state['env_np_random']

    def reset(self, **kwargs):

        if self._prev_saved_state is not None and self._prev_saved_state['reset_state']:
            init_state = self._prev_saved_state['env_state']
            self._prev_saved_state['reset_state'] = False
            prev_count = self.env._elapsed_steps
            state = super(PickleableEnv, self).reset(init_state=init_state, **kwargs)
            self.env._elapsed_steps = prev_count
            if 'last_u' in self._prev_saved_state and hasattr(self, 'last_u'):
                self.last_u = self._prev_saved_state['last_u']
            return state
        else:
            return super(PickleableEnv, self).reset(**kwargs)

class StepMonitor(Monitor):
    def __init__(self, env, directory, env_name='acrobot', pickleable=True, **kwargs):
        super(StepMonitor, self).__init__(env, directory, **kwargs)
        self._env_name = env_name
        self._reset_count = 0
        self._original_prefix = self.file_prefix
        self.file_prefix = self._original_prefix + f"_{self._reset_count}"
        self._previous_file_prefix = None
        self._prev_saved_state = None
        self._experiment_dir = directory
        self._start = True
        self._pickleable = pickleable

    def reset_video_recorder(self):
        super(StepMonitor, self).reset_video_recorder()
        steps_taken = 0
        if hasattr(self.env, '_elapsed_steps'):
            steps_taken = self.env._elapsed_steps
        elif hasattr(self.env, 'step_count'):
            steps_taken = self.env.step_count
        # add num_steps to beginning of video
        if steps_taken > 0 and not self._start:
            for fname in os.listdir(self._experiment_dir):
                if fname.startswith(self._previous_file_prefix):
                    full_src_fname = os.path.join(self._experiment_dir, fname)
                    full_dst_fname = os.path.join(self._experiment_dir, f'{steps_taken}_{fname}')
                    os.rename(full_src_fname, full_dst_fname)
        self._reset_count += 1
        self._previous_file_prefix = self.file_prefix
        self.file_prefix = self._original_prefix + f"_{self._reset_count}"

    def step(self, action):
        self._start = False
        return super(StepMonitor, self).step(action)

    def __getstate__(self):
        state = {
            'reset_count': self._reset_count,
            'original_prefix': self._original_prefix,
            'env_np_random': self.unwrapped.np_random,
            'reset_state': True
        }
        if self._pickleable:
            state['env_state'] = self.unwrapped.state

        if hasattr(self, 'last_u'):
            state['last_u'] = self.last_u

        return state


    def __setstate__(self, state):
        self._reset_count = state['reset_count']
        self._original_prefix = state['original_prefix']
        self._prev_saved_state = state

    def reinitialize(self, env, directory, video_callable=None):
        assert self._prev_saved_state is not None
        super(StepMonitor, self).__init__(env, directory, video_callable=video_callable)

        self._original_prefix = self.file_prefix
        self.file_prefix = self._original_prefix + f"_{self._reset_count}"
        self._experiment_dir = directory
        self.unwrapped.np_random = self._prev_saved_state['env_np_random']

    def reset(self, **kwargs):

        if self._prev_saved_state is not None and self._prev_saved_state['reset_state']:
            init_state = self._prev_saved_state['env_state']
            self._prev_saved_state['reset_state'] = False
            prev_count = self.env._elapsed_steps
            state = super(StepMonitor, self).reset(init_state=init_state, **kwargs)
            self.env._elapsed_steps = prev_count
            if 'last_u' in self._prev_saved_state and hasattr(self, 'last_u'):
                self.last_u = self._prev_saved_state['last_u']
            return state
        else:
            return super(StepMonitor, self).reset(**kwargs)

class ContinuablePendulumEnv(PendulumEnv):
    def __init__(self, *args, **kwargs):
        super(ContinuablePendulumEnv, self).__init__(*args, **kwargs)

    def reset(self, init_state=None):
        low = np.array([3 * np.pi / 4, -1])
        # low = np.array([0, -1])
        # high = np.array([0.1, 1])
        high = np.array([np.pi, 1])
        if init_state is None:
            self.state = self.np_random.uniform(low=low, high=high)
            sign = -1 if self.np_random.uniform() > 0.5 else 1
            self.state[0] *= sign
        else:
            self.state = init_state

        self.last_u = None
        return self._get_obs()

    def step(self,u):
        obs, reward, terminal, info = super(ContinuablePendulumEnv, self).step(u)
        # abs_state_angle = np.abs(self.state[0])
        cos_state = np.cos(self.state[0])
        # done = abs_state_angle < np.pi / 4
        done = cos_state > 0.75

        if done:
            print("here")
            reward = 0
            self.done = done
            terminal = done

        return obs, reward, terminal, info

class ContinuableAcrobotEnv(AcrobotEnv):
    def __init__(self, *args, **kwargs):
        super(ContinuableAcrobotEnv, self).__init__(*args, **kwargs)

    def reset(self, init_state=None):
        if init_state is None:
            self.state = self.np_random.uniform(low=-0.1, high=0.1, size=(4,))
        else:
            self.state = init_state

        return self._get_ob()

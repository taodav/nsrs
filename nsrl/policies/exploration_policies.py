import torch
import copy
import numpy as np

from .EpsilonGreedyPolicy import EpsilonGreedyPolicy
from nsrl.helper.pytorch import device

class RewardArgmaxPolicy(EpsilonGreedyPolicy):
    def __init__(self, learning_algo, n_actions, random_state, epsilon_start=0):
        super(RewardArgmaxPolicy, self).__init__(learning_algo, n_actions, random_state, epsilon_start)

    def bestAction(self, state, mode=None, *args, **kwargs):
        for m in self.learning_algo.all_models: m.eval()

        R = self.learning_algo.R if self.learning_algo._train_reward else None
        copy_state = copy.deepcopy(state)  # Required because of the "hack" below
        state_tensor = torch.tensor(copy_state[0], dtype=torch.float).to(device)
        dataset = kwargs.get('dataset', None)
        if dataset is None:
            raise Exception()
        with torch.no_grad():
            abstr_state = self.learning_algo.encoder(state_tensor)

            all_prev_obs = torch.tensor(dataset.observationsMatchingBatchDim()[0], dtype=torch.float).to(device)
            all_prev_states = self.learning_algo.encoder(all_prev_obs)

            scores = self.learning_algo.intrRewards_planning(abstr_state, self.learning_algo.transition, all_prev_states, R=R)

        return np.argmax(scores, axis=-1), np.max(scores, axis=-1)


class QArgmaxPolicy(EpsilonGreedyPolicy):
    def __init__(self, learning_algo, n_actions, random_state, epsilon_start=0):
        super(QArgmaxPolicy, self).__init__(learning_algo, n_actions, random_state, epsilon_start)

    def bestAction(self, state, mode=None, *args, **kwargs):
        for m in self.learning_algo.all_models: m.eval()

        with torch.no_grad():
            copy_state = copy.deepcopy(state)  # Required because of the "hack" below
            state_tensor = torch.tensor(copy_state[0], dtype=torch.float).to(device)
            q_vals = self.learning_algo.qValues(state_tensor).squeeze(0).cpu().numpy()
        return np.argmax(q_vals, axis=-1), np.max(q_vals, axis=-1)


class MCPolicy(EpsilonGreedyPolicy):
    def __init__(self, learning_algo, n_actions, random_state, depth=1, epsilon_start=0):
        super(MCPolicy, self).__init__(learning_algo, n_actions, random_state, epsilon_start)
        self._depth = depth

    def bestAction(self, state, mode=None, *args, **kwargs):
        for m in self.learning_algo.all_models: m.eval()

        with torch.no_grad():
            R = self.learning_algo.R if self.learning_algo._train_reward else None
            copy_state = copy.deepcopy(state)  # Required because of the "hack" below
            state_tensor = torch.tensor(copy_state[0], dtype=torch.float).to(device)
            dataset = kwargs.get('dataset', None)
            if dataset is None:
                raise Exception()
            abstr_state = self.learning_algo.encoder(state_tensor)
            all_prev_obs = torch.tensor(dataset.observationsMatchingBatchDim()[0], dtype=torch.float).to(device)
            all_prev_states = self.learning_algo.encoder(all_prev_obs)
            scores = self.learning_algo.novelty_d_step_planning(abstr_state, self.learning_algo.Q, self.learning_algo.transition, all_prev_states, R=R, d=self._depth,
                                                  b=self.n_actions)

        return np.argmax(scores, axis=-1), np.max(scores, axis=-1)


class MCRewardPolicy(EpsilonGreedyPolicy):
    def __init__(self, learning_algo, n_actions, random_state, depth=1, epsilon_start=0):
        super(MCRewardPolicy, self).__init__(learning_algo, n_actions, random_state, epsilon_start)
        self._depth = depth

    def bestAction(self, state, mode=None, *args, **kwargs):
        for m in self.learning_algo.all_models: m.eval()

        with torch.no_grad():
            R = self.learning_algo.R if self.learning_algo._train_reward else None
            copy_state = copy.deepcopy(state)  # Required because of the "hack" below
            state_tensor = torch.tensor(copy_state[0], dtype=torch.float).to(device)
            dataset = kwargs.get('dataset', None)
            if dataset is None:
                raise Exception()

            # This Q returns all 0s for all predicted Q values
            class Q_zeros:
                @staticmethod
                def predict(abstr_reps):
                    return torch.zeros((abstr_reps.shape[0], self.n_actions))

            abstr_state = self.learning_algo.encoder(state_tensor)
            all_prev_obs = torch.tensor(dataset.observationsMatchingBatchDim()[0], dtype=torch.float).to(device)
            all_prev_states = self.learning_algo.encoder(all_prev_obs)
            scores = self.learning_algo.novelty_d_step_planning(abstr_state, Q_zeros, self.learning_algo.transition, all_prev_states, R=R, d=self._depth,
                                                  b=self.n_actions)

        return np.argmax(scores, axis=-1), np.max(scores, axis=-1)

class BootstrapDQNPolicy(EpsilonGreedyPolicy):
    def __init__(self, learning_algo, n_actions, random_state, epsilon_start=0):
        super(BootstrapDQNPolicy, self).__init__(learning_algo, n_actions, random_state, epsilon_start)
        self.idx = 0
        self.head_num = self.learning_algo.Q.n_heads

    def sample_head(self):
        self.idx = np.random.randint(self.head_num)

    def bestAction(self, state, mode=None, *args, **kwargs):
        for m in self.learning_algo.all_models: m.eval()

        with torch.no_grad():
            copy_state = copy.deepcopy(state)  # Required because of the "hack" below
            state_tensor = torch.tensor(copy_state[0], dtype=torch.float).to(device)

            abstr_state = self.learning_algo.encoder(state_tensor)
            # Refer to BootstrappedQFunction here
            scores = self.learning_algo.Q(abstr_state, [self.idx])[0].cpu().numpy()[0]

        return np.argmax(scores, axis=-1), np.max(scores, axis=-1)



import json
import os
import copy
import numpy as np
import torch
import json
from nsrl.experiment.base_controllers import Controller
from nsrl.helper.exploration import calculate_scores
from nsrl.helper.knn import ranked_avg_knn_scores, batch_count_scaled_knn
from nsrl.helper.pytorch import device, calculate_large_batch


class ExplorationMetricController(Controller):
    def __init__(self, evaluate_on='action', periodicity=1, reset_every='none',
                 env_name='default', experiment_dir=None, baseline_file=None,
                 hyperparams=None, reload_dataset=None, **kwargs):
        """
        Controller for PLOTTING exploration metric. Requires Visdom.
        :param evaluate_on:
        :param periodicity:
        :param reset_every:
        """
        super(ExplorationMetricController, self).__init__(**kwargs)

        self._periodicity = periodicity
        self._baseline_file = baseline_file

        self._on_action = 'action' == evaluate_on
        self._on_episode = 'episode' == evaluate_on
        self._on_epoch = 'epoch' == evaluate_on
        self._env_name = env_name
        self._experiment_dir = experiment_dir
        self._exp_factor = []
        self._ratio_visited = []
        if not self._on_action and not self._on_episode and not self._on_epoch:
            self._on_action = True

        if reload_dataset is not None:
            self._reload_dataset(reload_dataset)

        self._reset_on_episode = 'episode' == reset_every
        self._reset_on_epoch = 'epoch' == reset_every
        self._ep_num = -1
        self._hyperparams = hyperparams


    def _plot_baseline(self, agent):
        if self._baseline_file is not None:
            # If a baseline data is given, we overlay it.
            with open(self._baseline_file, 'r') as f:
                baseline = json.load(f)
                exp_factor = np.array([l for l in baseline['exploration_factors'] if l])
                avg_exp_factor = np.average(exp_factor, axis=0)

                agent._plotter.plot("exploration_factor",
                                    np.arange(0, len(avg_exp_factor)), avg_exp_factor,
                                    "Exploration Factor",
                                    ymin=0, ymax=1, name='baseline')

                ratio_visited = np.array([l for l in baseline['ratios_visited'] if l])
                avg_ratio_visited = np.average(ratio_visited, axis=0)
                agent._plotter.plot("states_visited",
                                    np.arange(0, len(avg_ratio_visited)), avg_ratio_visited,
                                    "Ratio of states visited",
                                    ymin=0, ymax=1, name='baseline')

    def onStart(self, agent):
        if (self._active == False):
            return

        self._reset(agent)

    def onEpisodeEnd(self, agent, terminal_reached, reward):
        if (self._active == False):
            return

        if self._reset_on_episode:
           self. _reset(agent)
        elif self._on_episode:
            self._update(agent)

    def onEpochEnd(self, agent):
        if (self._active == False):
            return

        if self._reset_on_epoch:
            self._reset(agent)
        elif self._on_epoch:
            self._update(agent)

    def onActionTaken(self, agent):
        if (self._active == False):
            return

        if self._on_action:
            self._update(agent)

    def _reset(self, agent):
        self._count = 0
        self._exp_factor.append([])
        self._ratio_visited.append([])

        self._plot_baseline(agent)

        if agent._dataset.n_elems > len(self._exp_factor[self._ep_num]) and len(self._exp_factor[self._ep_num]) == 0:
            self._plot_dataset(agent)
            self._count = agent._dataset.n_elems

        self._ep_num += 1

    def _plot_dataset(self, agent):
        all_observations = agent._dataset.observationsMatchingBatchDim()[0]
        all_positions = [(y.item(), x.item()) for y, x in zip(*np.where(all_observations == 0.5)[1:])]
        unique_counts = []
        for i, pos in enumerate(all_positions):
            if i == 0:
                unique_counts.append(1)
            else:
                unique_counts.append(unique_counts[i - 1])
                if pos not in all_positions[:i]:
                    unique_counts[i] += 1

        # self._exp_factor[self._ep_num] = [c / (i + 1) for i, c in enumerate(unique_counts)]

        y = np.array(self._exp_factor[self._ep_num])
        x = np.arange(len(self._exp_factor[self._ep_num]))
        agent._plotter.plot("exploration_factor",
                            x, y,
                            "Exploration Factor", ymin=0, ymax=1)

        if hasattr(agent._environment, "_size_maze"):
            ys, xs = np.nonzero(agent._environment._map == 0.0)
            total_possible_states = len(ys)
            self._ratio_visited[self._ep_num] = [c / total_possible_states for c in unique_counts]
            y_tps = np.array(self._ratio_visited[self._ep_num])
            agent._plotter.plot("states_visited",
                                x, y_tps,
                                "Ratio of states visited",
                                ymin=0, ymax=1)

    def _update(self, agent):

        self._count += 1

        if self._periodicity <= 1 or self._count % self._periodicity == 0:

            all_observations = agent._dataset.observations()[0]
            if all_observations.shape[0] < 1:
                return
            unique_observations = np.unique(all_observations, axis=0)
            exp_factor = unique_observations.shape[0] / all_observations.shape[0]
            self._exp_factor[self._ep_num].append(exp_factor)
            x = np.array([self._count])
            y = np.array([exp_factor])
            agent._plotter.plot("exploration_factor",
                                x, y,
                                "Exploration Factor",
                                ymin=0, ymax=1)

            if hasattr(agent._environment, "_size_maze"):
                ys, xs = np.nonzero(agent._environment._map == 0.0)
                total_possible_states = len(ys)
                ratio = unique_observations.shape[0] / total_possible_states
                self._ratio_visited[self._ep_num].append(ratio)
                y_tps = np.array([ratio])
                agent._plotter.plot("states_visited",
                                    x, y_tps,
                                    "Ratio of states visited",
                                    ymin=0, ymax=1)

    def onEnd(self, agent):
        exp_factor = np.array([l for l in self._exp_factor if l])
        avg_exp_factor = np.average(exp_factor, axis=0)

        agent._plotter.plot("average exploration factor",
                            np.arange(0, len(avg_exp_factor)), avg_exp_factor,
                            "Average exploration factor over %d episodes" % exp_factor.shape[0],
                            ymin=0, ymax=1)

        ratio_visited = np.array([l for l in self._ratio_visited if l])
        avg_ratio_visited = np.average(ratio_visited, axis=0)

        agent._plotter.plot("average ratios visited",
                            np.arange(0, len(avg_exp_factor)), avg_ratio_visited,
                            "Average ratio of states visited over %d episodes" % ratio_visited.shape[0],
                            ymin=0, ymax=1)

        record = {
            'exploration_factors': self._exp_factor,
            'ratios_visited': self._ratio_visited
        }
        if self._hyperparams is not None:
            record['hyperparameters'] = self._hyperparams

        filename = os.path.join(self._experiment_dir, 'results.json')
        with open(filename, 'w') as f:
            json.dump(record, f)


class RewardController(Controller):
    def __init__(self, evaluate_on='train_loop', periodicity=1):
        super(RewardController, self).__init__()
        self._on_train_loop = 'train_loop' == evaluate_on
        self._on_train_step = 'train_step' == evaluate_on
        self._before_action = 'before_action' == evaluate_on
        self._on_action = 'action' == evaluate_on
        self._on_episode = 'episode' == evaluate_on
        self._on_epoch = 'epoch' == evaluate_on

        self._periodicity = periodicity
        self._count = 0

    def onActionTaken(self, agent):
        if self._on_action:
            if self._count % self._periodicity == 0:
                self._update(agent)
            self._count += 1

    def onActionChosen(self, agent, action):
        if self._before_action:
            if self._count % self._periodicity == 0:
                self._update(agent)
            self._count += 1

    def onEpochEnd(self, agent):
        if self._on_epoch:
            if self._count % self._periodicity == 0:
                self._update(agent)
            self._count += 1

    def onTrainLoopTaken(self, agent):
        if self._on_train_loop:
            if self._count % self._periodicity == 0:
                self._update(agent)
            self._count += 1

    def repopulate_rewards(self, agent):
        self._update(agent)

    def _update(self, agent):
        raise NotImplementedError


class NoveltyRewardController(RewardController):
    def __init__(self, evaluate_on='train_loop', periodicity=1,
                 metric_func=calculate_scores,
                 score_func=ranked_avg_knn_scores, k=10, knn=batch_count_scaled_knn,
                 secondary=False):
        super(NoveltyRewardController, self).__init__(evaluate_on=evaluate_on, periodicity=periodicity)

        self._metric_func = metric_func
        self._k = k
        self._score_func = score_func
        self._knn = knn
        self._secondary = secondary

    def _update(self, agent):
        # Now we have to calculate intrinsic rewards
        for m in agent._learning_algo.all_models: m.eval()

        all_prev_state = agent._dataset.observationsMatchingBatchDim()[0]
        intr_rewards = self._metric_func(all_prev_state,
                                         all_prev_state,
                                         agent._learning_algo.encoder,
                                         dist_score=self._score_func,
                                         k=self._k, knn=self._knn)
        # UPDATE HTIS TO TAKE INTO ACCOUNT NON INTRINSIC REWARDS

        # reward clipping for preventing divergence
        # intr_rewards = np.clip(intr_rewards, -1, 1)

        # s_t, a_t, r_t (where r_t is intr_reward of s_{t+1})
        agent._dataset.updateRewards(intr_rewards[1:], np.arange(0, agent._dataset.n_elems - 1), secondary=self._secondary)

        # we still need to calculate most recent reward.
        latest_state = np.array(agent._environment.observe())
        if len(latest_state.shape) != len(all_prev_state.shape):
            latest_obs = latest_state
            obs_per_state = all_prev_state.shape[1]
            n_to_fill = obs_per_state - 1
            n_prev_obs = agent._dataset.observations()[0][-n_to_fill:]
            latest_state = np.expand_dims(np.concatenate((n_prev_obs, latest_obs), axis=0), axis=0)

        latest_obs_intr_reward = self._metric_func(latest_state,
                                                  all_prev_state,
                                                  agent._learning_algo.encoder,
                                                  dist_score=self._score_func,
                                                  k=self._k, knn=self._knn)
        # latest_obs_intr_reward = np.clip(latest_obs_intr_reward, -1, 1)
        agent._dataset.updateRewards(latest_obs_intr_reward, agent._dataset.n_elems - 1, secondary=self._secondary)

class HashStateCounterController(RewardController):
    def __init__(self, plotter, evaluate_on='action', periodicity=1, granularity=32,
                 input_dims=(1, 64, 64), **kwargs):
        self._periodicity = periodicity
        self._count = 0
        super(HashStateCounterController, self).__init__(evaluate_on=evaluate_on, periodicity=periodicity)
        self._granularity = granularity  # size of binary code
        # Get input_dims from self._environment.inputDimensions()[0]
        self._A = np.random.normal(size=(self._granularity, np.prod(input_dims)))
        self.plotter = plotter
        self._unique_state_count = [0]
        self._count_table = {}

    def onStart(self, agent):
        all_obs = agent._dataset.observationsMatchingBatchDim()[0]
        indices_to_update = [0]
        for i, ob in enumerate(all_obs[1:], 1):
            hashed_ob = self.calc_hash(ob)
            to_add = self._unique_state_count[-1]
            if hashed_ob not in self._count_table:
                self._count_table[hashed_ob] = 0
                to_add += 1
            self._unique_state_count.append(to_add)
            self._count_table[hashed_ob] += 1
            indices_to_update.append(i)
            self._count += 1

        if indices_to_update:
            self.plotter.plot('hashed_unique_state_counts', indices_to_update, self._unique_state_count)

    def onEnd(self, agent):
        self.plotter.plot_text('ending', f'Environment completed after {self._count} steps')

    def calc_hash(self, obs):
        A_g = np.matmul(self._A, obs.flatten())
        hash_seq = np.sign(A_g).astype(int)
        zero_mask = (hash_seq == 0).astype(int)
        hash_seq += zero_mask

        return str(hash_seq)

    def _update(self, agent):
        all_obs = agent._dataset.observationsMatchingBatchDim()[0]
        latest_obs = agent._environment.observe()[0]
        if len(all_obs.shape) == 4:
            second_to_last = all_obs[-1]
            to_attach = second_to_last[1:]
            latest_obs = np.concatenate((to_attach, latest_obs[None, :, :]), axis=0)
        hashed_obs = self.calc_hash(latest_obs)
        to_add = 0
        if hashed_obs not in self._count_table:
            to_add = 1
            self._count_table[hashed_obs] = 0
        self._count_table[hashed_obs] += 1
        self._unique_state_count.append(self._unique_state_count[-1] + to_add)

        self.plotter.plot('hashed_unique_state_counts', [self._count], [self._unique_state_count[-1]], 'Hashed Unique State Counts')

class HashCountRewardController(RewardController):
    def __init__(self, evaluate_on='action', periodicity=1, granularity=32,
                 input_dims=(1, 64, 64), secondary=False, discrete=False, **kwargs):
        super(HashCountRewardController, self).__init__(evaluate_on=evaluate_on, periodicity=periodicity)
        self._granularity = granularity  # size of binary code
        self._bonus_coeff = 0.01 * (256 / self._granularity)
        if discrete:
            self._bonus_coeff = 1
        # self._A = np.random.normal(size=(self._granularity, self._learning_algo.internal_dim))
        # Get input_dims from self._environment.inputDimensions()[0]
        self._A = np.random.normal(size=(self._granularity, np.prod(input_dims)))
        self._count_table = {}
        self._secondary = secondary
        self._discrete = discrete

        # FOR DEBUGGING
        self._all_latest_obs = []
        self._all_obs = []

    def onStart(self, agent):
        all_obs = agent._dataset.observationsMatchingBatchDim()[0]

        hashed_obs = []
        indices_to_update = []
        for i, ob in enumerate(all_obs[1:]):
            hashed_ob = self.calc_hash(ob)
            hashed_obs.append(hashed_ob)
            if hashed_ob not in self._count_table:
                self._count_table[hashed_ob] = 0
            self._count_table[hashed_ob] += 1
            indices_to_update.append(i)
        rewards = []
        for hob in hashed_obs:
            rewards.append(self._bonus_coeff / np.sqrt(self._count_table[hob]))
        agent._dataset.updateRewards(rewards, indices_to_update, secondary=self._secondary)

    def calc_hash(self, obs):
        if self._discrete:
            return np.array2string(obs.astype(np.half).flatten())
        else:
            A_g = np.matmul(self._A, obs.flatten())
            hash_seq = np.sign(A_g).astype(int)
            zero_mask = (hash_seq == 0).astype(int)
            hash_seq += zero_mask

            return str(hash_seq)

    def _update(self, agent):
        all_obs = agent._dataset.observationsMatchingBatchDim()[0]
        # hacky as shit
        if len(all_obs) == 1:
            # we need to first count the first state
            hashed_first_ob = self.calc_hash(all_obs[0])
            self._count_table[hashed_first_ob] = 1

        latest_obs = agent._environment.observe()[0]
        self._all_obs.append(copy.deepcopy(latest_obs))
        if len(all_obs.shape) == 4:
            second_to_last = all_obs[-1]
            to_attach = second_to_last[1:]
            latest_obs = np.concatenate((to_attach, latest_obs[None, :, :]), axis=0)
        self._all_latest_obs.append(latest_obs)
        hashed_obs = self.calc_hash(latest_obs)
        # FIGURE THIS OUT FOR ACROBOT
        if hashed_obs not in self._count_table:
            self._count_table[hashed_obs] = 0

        self._count_table[hashed_obs] += 1
        next_states = np.concatenate((all_obs[1:],latest_obs[np.newaxis]), axis=0)
        # idx_to_update = [i for i, obs in enumerate(all_obs[1:]) if hashed_obs == self.calc_hash(obs)]
        idx_to_update = np.arange(len(all_obs))
        # idx_to_update = np.array(idx_to_update)
        # THIS MIGHT NEED TO BE REFACTORED
        all_rewards = []
        for i, s in enumerate(next_states):
            hob = self.calc_hash(s)
            all_rewards.append(self._bonus_coeff / np.sqrt(self._count_table[hob]))

        agent._dataset.updateRewards(all_rewards, idx_to_update, secondary=self._secondary)

class CountBasedRewardController(RewardController):
    def __init__(self, evaluate_on='action', periodicity=1, bonus=1, hash_func=None, secondary=False, **kwargs):
        super(CountBasedRewardController, self).__init__(evaluate_on=evaluate_on, periodicity=periodicity)
        self._hash_func = hash_func
        self._bonus = bonus
        self._counts = {}
        self._secondary = secondary

    def _update(self, agent):
        """
        increment counts and update rewards
        :param agent:
        :return:
        """
        for m in agent._learning_algo.all_models: m.eval()

        all_obs = agent._dataset.observations()[0]

        latest_obs = agent._environment.observe()[0]
        hashed_obs = self._hash_func(latest_obs)
        if hashed_obs not in self._counts:
            self._counts[hashed_obs] = 0

        self._counts[hashed_obs] += 1


        idx_to_update = [i for i, obs in enumerate(all_obs[1:]) if hashed_obs == self._hash_func(obs)]
        idx_to_update = np.array(idx_to_update)
        if len(idx_to_update) > 0:
            new_reward = self._bonus / np.sqrt(self._counts[hashed_obs])
            rewards = np.repeat(new_reward, len(idx_to_update))
            agent._dataset.updateRewards(rewards, idx_to_update, secondary=self._secondary)

class TransitionLossRewardController(RewardController):
    def __init__(self, evaluate_on='train_loop', periodicity=1, secondary=False, **kwargs):
        super(TransitionLossRewardController, self).__init__(evaluate_on=evaluate_on, periodicity=periodicity)

        self._secondary = secondary

    def _update(self, agent):
        for m in agent._learning_algo.all_models: m.eval()

        all_obs = agent._dataset.observationsMatchingBatchDim()[0]
        states = torch.tensor(all_obs, dtype=torch.float).to(device)
        latest_obs = agent._environment.observe()[0][None, :, :]
        if len(all_obs.shape) == 4:
            second_to_last = all_obs[-1]
            to_attach = second_to_last[1:]
            latest_obs = np.concatenate((to_attach, latest_obs), axis=0)
            latest_obs = latest_obs[None, :, :, :]

        next_states = torch.tensor(np.concatenate((all_obs[1:],latest_obs), axis=0), dtype=torch.float, requires_grad=False).to(device)

        actions = agent._dataset.actions()
        one_hot_actions = np.zeros((actions.shape[0], agent._environment.nActions()))
        one_hot_actions[np.arange(len(actions)), actions] = 1
        one_hot_actions = torch.tensor(one_hot_actions, requires_grad=False).float().to(device)

        encoder = agent._learning_algo.encoder
        transition = agent._learning_algo.transition

        with torch.no_grad():
            abstr_states = encoder(states)
            target_next_states = encoder(next_states)

            transition_input = torch.cat((abstr_states, one_hot_actions), dim=-1)
            abstr_next_states = transition(transition_input)
            squared_diff = torch.sum((abstr_next_states - target_next_states) ** 2, dim=-1)
            rewards = squared_diff.cpu().numpy()

        idx_to_update = list(range(len(all_obs)))
        agent._dataset.updateRewards(rewards, idx_to_update, secondary=self._secondary)


class RNDRewardController(RewardController):
    def __init__(self, evaluate_on='action', periodicity=1,
                 secondary=False):
        super(RNDRewardController, self).__init__(evaluate_on=evaluate_on, periodicity=periodicity)
        self._secondary = secondary

    @torch.no_grad()
    def _update(self, agent):
        for m in agent._learning_algo.all_models: m.eval()
        all_prev_state = agent._dataset.observationsMatchingBatchDim()[0]
        states = torch.tensor(all_prev_state, dtype=torch.float, device=device)
        abstr_states = calculate_large_batch(agent._learning_algo.encoder, states)
        predictor = agent._learning_algo.rnd_network.predictor
        target = agent._learning_algo.rnd_network.target
        pred_features = calculate_large_batch(predictor, abstr_states)
        target_features = calculate_large_batch(target, abstr_states)

        intrinsic_reward = (target_features - pred_features).pow(2).sum(1) / 2
        intrinsic_reward = intrinsic_reward.cpu().numpy()

        idx_to_update = list(range(len(all_prev_state)))
        agent._dataset.updateRewards(intrinsic_reward, idx_to_update, secondary=self._secondary)


class AbstractRepPlottingController(Controller):
    def __init__(self, plotter, evaluate_on='train_loop', periodicity=3, **kwargs):
        super(AbstractRepPlottingController, self).__init__(**kwargs)
        self._on_train_loop = 'train_loop' == evaluate_on
        self._on_train_step = 'train_step' == evaluate_on
        self._on_action = 'action' == evaluate_on
        self._on_episode = 'episode' == evaluate_on
        self._on_epoch = 'epoch' == evaluate_on

        self._periodicity = periodicity

        self._plotter = plotter
        self._limit_history = kwargs.get('limit_history', -1)
        self._skip_first = kwargs.get('skip_first', 0)

    def onActionTaken(self, agent):
        if self._on_action:
            if self._count > self._skip_first and self._count % self._periodicity == 0:
                self._update(agent)
            self._count += 1

    def onTrainLoopTaken(self, agent):
        if self._on_train_loop:
            if self._count > self._skip_first and self._count % self._periodicity == 0:
                self._update(agent)
            self._count += 1

    def onTrainStepTaken(self, agent):
        if self._on_train_step:
            if self._count > self._skip_first and self._count % self._periodicity == 0:
                self._update(agent)
            self._count += 1

    def onEnd(self, agent):
        self._update(agent)

    def plot_idx(self, agent, idx):
        obs = agent._dataset.observations()[0][idx][3].astype('float64')
        agent._environment.plot_state(obs)

    def _update(self, agent):
        fig = agent._environment.summarizePerformance(agent._dataset,
                                                      agent._learning_algo,
                                                      n_observations=self._limit_history)

        self._plotter.plot_plotly_fig("abstr_rep_step_%d" % self._count, fig,
                                      title_name="abstr reps_step %d" % self._count)


class RNNQHistoryController(Controller):
    def __init__(self, evaluate_on='action_taken', periodicity=1):
        super(RNNQHistoryController, self).__init__()
        self._on_train_loop = 'train_loop' == evaluate_on
        self._on_train_step = 'train_step' == evaluate_on
        self._on_action_taken = 'action_taken' == evaluate_on
        self._on_episode = 'episode' == evaluate_on
        self._on_epoch = 'epoch' == evaluate_on

        self._periodicity = periodicity

    def onActionTaken(self, agent):
        if self._on_action_taken:
            self._update(agent)

    def _update(self, agent):
        all_obs = agent._dataset.observations()[0]
        agent._learning_algo.Q.set_history(all_obs)
        agent._learning_algo.Q_target.set_history(all_obs)

class ExtrinsicRewardPlottingController(Controller):
    def __init__(self, plotter):
        super(ExtrinsicRewardPlottingController, self).__init__()
        self._plotter = plotter
        self._count = 0

    def onActionTaken(self, agent):
        reward = agent._dataset.rewards()[-1]
        self._plotter.plot("extrinsic_rewards", np.array([self._count]), np.array([reward]), title_name="Extrinsic Rewards")
        self._count += 1

class UniqueStateCounterController(Controller):
    def __init__(self, plotter, evaluate_on='action', periodicity=1, **kwargs):
        super(UniqueStateCounterController, self).__init__(**kwargs)
        self._on_train_loop = 'train_loop' == evaluate_on
        self._on_train_step = 'train_step' == evaluate_on
        self._on_action = 'action' == evaluate_on
        self._on_episode = 'episode' == evaluate_on
        self._on_epoch = 'epoch' == evaluate_on

        self._data = []
        self._states = set()
        self._count = 0
        self._num_uniques = 0
        self._periodicity = periodicity
        self._plotter = plotter

    def onStart(self, agent):
        trajectory = agent._environment._trajectory
        for t in trajectory:
            s = str(t)
            if s not in self._states:
                self._states.add(s)
                self._num_uniques += 1
            self._data.append(self._num_uniques)
            self._count += 1

        self._plotter.plot('unique_state_counter', list(range(self._count)), self._data, title_name='Unique State Counter')

    def onActionTaken(self, agent):
        if self._on_action:
            if self._count > 0 and self._count % self._periodicity == 0:
                self._update(agent)
            self._count += 1

    def onEnd(self, agent):
        self._plotter.plot_text('ending', f'Environment completed after {self._count} steps')

    def _update(self, agent):
        latest = agent._environment._trajectory[-1]
        sl = str(latest)
        if sl not in self._states:
            self._states.add(sl)
            self._num_uniques += 1
        self._data.append(self._num_uniques)

        self._plotter.plot('unique_state_counter', [self._count], [self._num_uniques], title_name='Unique State Counter')


class MapPlottingController(Controller):
    def __init__(self, plotter, evaluate_on='train_loop', periodicity=3,
                 metric_func=calculate_scores, k=10, learn_representation=True,
                 reward_type='novelty_reward', train_q=True,
                 internal_dim=2, plot_quiver=True, **kwargs):
        super(MapPlottingController, self).__init__(**kwargs)
        self._on_train_loop = 'train_loop' == evaluate_on
        self._on_train_step = 'train_step' == evaluate_on
        self._on_action = 'action' == evaluate_on
        self._on_episode = 'episode' == evaluate_on
        self._on_epoch = 'epoch' == evaluate_on

        self._periodicity = periodicity
        self._metric_func = metric_func
        self._reward_type = reward_type
        self._learn_representation = learn_representation
        self._k = k
        self._train_q = train_q
        self._internal_dim = internal_dim
        self._plot_quiver = plot_quiver

        self._plotter = plotter
    # def onStart(self, agent):
    #     print("starting")

    def onTrainLoopTaken(self, agent):
        if self._on_train_loop:
            if self._count > 0 and self._count % self._periodicity == 0:
                self._update(agent)
            self._count += 1

    def onTrainStepTaken(self, agent):
        if self._on_train_step:
            if self._count > 0 and self._count % self._periodicity == 0:
                self._update(agent)
            self._count += 1

    def onStart(self, agent):
        print("here")

    def onActionTaken(self, agent):
        if self._on_action:
            if self._count > 0 and self._count % self._periodicity == 0:
                self._update(agent)
            self._count += 1

    def _update(self, agent):
        for m in agent._learning_algo.all_models: m.eval()

        all_prev_obs = agent._dataset.observationsMatchingBatchDim()[0]
        # intr_rewards = agent._dataset.rewards()[:-1]
        all_possible_obs = all_prev_obs
        if hasattr(agent._environment, 'getAllPossibleStates'):
            all_possible_obs = agent._environment.getAllPossibleStates()
        # all_possible_obs = all_prev_obs


        # Now we might want to map the Q values of the given states.
        example = agent._environment._map
        borders = example == 1
        intr_rewards_map = np.zeros(example.shape) + borders.astype(int)
        vals_map = np.zeros(example.shape) + borders.astype(int)
        q_vals_map = np.zeros(example.shape + (4,))
        state_count_map = np.zeros_like(example) + borders.astype(int)

        q_vals = np.zeros((len(all_possible_obs), agent._environment.nActions()))

        if self._train_q:
            with torch.no_grad():
                if agent._bootstrap_q:
                    Q = agent._learning_algo.Q
                    n_heads = Q.n_heads
                    copy_state = copy.deepcopy(all_possible_obs)  # Required!
                    all_possible_obs_tensor = torch.tensor(copy_state, dtype=torch.float).to(device)
                    all_possible_abstr = agent._learning_algo.encoder(all_possible_obs_tensor)
                    all_qs = torch.stack(Q(all_possible_abstr, list(range(n_heads))))
                    q_vals = torch.mean(all_qs, dim=0).cpu().detach().numpy()
                else:
                    q_vals = agent._learning_algo.qValues(all_possible_obs).cpu().detach().numpy()

        intr_rewards = np.zeros_like(q_vals)

        if self._reward_type == 'novelty_reward':
            intr_rewards = self._metric_func(all_possible_obs,
                                             all_prev_obs,
                                             agent._learning_algo.encoder,
                                             k=self._k)
            # intr_rewards = np.clip(intr_rewards, -1, 1)

        for r, q, s in zip(intr_rewards, q_vals, all_possible_obs):
            pos_y, pos_x = np.where(s == 0.5)
            if self._reward_type == 'novelty_reward':
                intr_rewards_map[pos_y, pos_x] = r

            if self._train_q:
                q_vals_map[pos_y, pos_x] = q

            max_vals = np.max(q, axis=-1)
            vals_map[pos_y, pos_x] = max_vals.item()


        for pos_y, pos_x in agent._environment._trajectory:
            state_count_map[pos_y, pos_x] += 1

        if self._reward_type == 'count_reward':
            intr_rewards_map = 1 / (np.sqrt(state_count_map))
            infs = np.isinf(intr_rewards_map)
            intr_rewards_map[infs] = 0

        heatmaps = [("counts step %d" % self._count, state_count_map)]

        if self._train_q:
            # for logging purposes
            q_up_map = q_vals_map[:, :, 0] + borders
            q_down_map = q_vals_map[:, :, 1] + borders
            q_left_map = q_vals_map[:, :, 2] + borders
            q_right_map = q_vals_map[:, :, 3] + borders

            # Now we make a window for each
            heatmaps = [
                ("Q values (up) for step %d" % self._count, q_up_map),
                ("Q values (down) for step %d" % self._count, q_down_map),
                ("Q values (left) for step %d" % self._count, q_left_map),
                ("Q values (right) for step %d" % self._count, q_right_map),
                ("State value function for step %d" % self._count, vals_map)
            ] + heatmaps
            if self._plot_quiver:
                self._plotter.plot_quiver("quiver_step_%d" % self._count,
                                          q_up_map, q_down_map,
                                          q_left_map, q_right_map,
                                          "Quiver map for step %d" % self._count)

        # if self._reward_type == 'novelty_reward':
        heatmaps.append(("Intrinsic reward map for step %d" % self._count, intr_rewards_map))

        cols = 4 if len(heatmaps) > 4 else len(heatmaps)
        self._plotter.plot_mapping_heatmap(
            "heatmaps_step_%d" % self._count,
            heatmaps,
            title_name="Heatmaps for step %d" % self._count,
            cols=cols)

        # Here we plot abstr representations
        if self._learn_representation and self._internal_dim < 4:
            # abstr_rep_fig, abstr_rep_plt = agent._environment.summarizePerformance(agent._dataset, agent._learning_algo)
            #
            # self._plotter.plot_mpl_fig("abstr_rep_step_%d" % self._count, abstr_rep_fig,
            #                            title_name="abstr reps_step %d" % self._count, replace=True)

            fig = agent._environment.summarizePerformance(agent._dataset, agent._learning_algo)
            self._plotter.plot_plotly_fig("abstr_rep_step_%d" % self._count, fig,
                                          title_name="abstr reps_step %d" % self._count)


        # self._log_trajectory(agent)

    def _log_trajectory(self, agent):
        trajectory = np.array(agent._environment._trajectory)
        x = trajectory[:, 0]
        y = trajectory[:, 1]
        agent._plotter.plot('trajectory step %d' % self._count,
                            x, y, xmin=0, xmax=agent._environment._size_maze,
                            ymin=0, ymax=agent._environment._size_maze,
                            title_name='Trajectory Plot for step %d' % self._count,
                            markers=True, linecolor=np.array([[255,0,0]]))


class LossPlottingController(Controller):
    def __init__(self, plotter, evaluate_on='train_step', sum_over=1000, periodicity=1, max_size=1000):
        super(LossPlottingController, self).__init__()
        self._on_train_loop = 'train_loop' == evaluate_on
        self._on_train_step = 'train_step' == evaluate_on
        self._on_action = 'action' == evaluate_on
        self._on_episode = 'episode' == evaluate_on
        self._on_epoch = 'epoch' == evaluate_on

        self._plotter = plotter
        self._periodicity = periodicity
        self._sum_over = sum_over
        self._count = 0
        self._buffer = dict(counts=[])
        self._max_size = max_size
        self._steps = 0

    def onTrainStepTaken(self, agent):
        if self._count % self._sum_over == 0 and self._on_train_step:
            self._update_buffer(agent)
            if self._count % (self._sum_over * self._periodicity) == 0:
                self._update()
        self._count += 1

    def onActionTaken(self, agent):
        self._steps += 1

    def _update_buffer(self, agent):
        for k, v in agent._all_losses.items():
            if v:
                if k not in self._buffer:
                    self._buffer[k] = []
                if len(self._buffer[k]) > self._sum_over + self._max_size:
                    self._buffer[k] = self._buffer[1:]
                self._buffer[k] += [np.mean(v[-self._sum_over:])]

        self._buffer['counts'] += [self._count // self._sum_over]

    def onEnd(self, agent):
        to_save = dict(
            total_steps=self._steps,
            # final_buffer=self._buffer
        )
        exp_dir = self._plotter.experiment_dir
        final_results_file = os.path.join(exp_dir, 'final_results.json')
        with open(final_results_file, 'w') as f:
            json.dump(to_save, f)

    def _update(self):
        counts = self._buffer['counts']
        del self._buffer['counts']
        self._plotter.plot_dict(counts, self._buffer)
        self._buffer = dict(counts=[])


def simple_hash_func(arr):
    """
    Simple hash function that finds the agent location (0.5) and returns that
    :param arr:
    :return: string of a tuple
    """
    pos_y, pos_x = np.where(arr == 0.5)
    return (pos_x.item(), pos_y.item())

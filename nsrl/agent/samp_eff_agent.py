import numpy as np
import copy
# import sys
# from torch.utils.data import DataLoader
# from pympler.asizeof import asizeof


from nsrl.agent import NeuralAgent
from nsrl.helper.pytorch import DataSet as IterativeDataset

class SEAgent(NeuralAgent):
    """
    Sample Efficient Exploration Agent
    Agent with a two-step training scheme
    First, we fit the abstract representations to everything in the dataset
    Next, we calculate the intrinsic rewards (and hence Q-Values) based on the
    calculate abstract representations.
    Finally, we fit our Q-values (Do we update our encoder here? Should
    we use value function instead?) w.r.t. abstract representations and taken
    actions. (We don't need to update other model-based components here??)
    """

    def __init__(self, environment, learning_algo, plotter, iters_per_update=1e4,
                 learn_representation=True, consec_dist=0.5, slack_ratio=4,
                 train_q=True, use_iterative_dataset=False,
                 epochs_per_train=20, train_nstep=1, add_single_cycle=False,
                 reward_learning="only_primary", **kwargs):
        super(SEAgent, self).__init__(environment, learning_algo, **kwargs)
        self._num_train_steps = kwargs.get('start_count', 0)

        self._learn_representation = learn_representation
        self._iters_per_update = iters_per_update

        self._plotter = plotter

        self._valid_size = 0.2
        self._consec_dist = consec_dist
        self._slack_ratio = slack_ratio
        self._train_q = train_q
        self._use_iterative_dataset = use_iterative_dataset
        self._epochs_per_train = epochs_per_train
        self._add_single_cycle = add_single_cycle
        self._reward_learning = reward_learning

        self._train_nstep = train_nstep

        replay_memory_size = kwargs.get('replay_memory_size')
        random_state = kwargs.get('random_state')

        self._populated = False
        self._initial_actions = [0, 3, 1, 2]

        self.gather_data = kwargs.get('gather_data', False)

        if self._use_iterative_dataset:
            self._dataset = IterativeDataset(environment, max_size=replay_memory_size,
                                             random_state=random_state,
                                             only_full_history=self._only_full_history)

        # Validation dataset for validation within training
        if self._use_iterative_dataset:
            self._valid_dataset = IterativeDataset(environment, max_size=replay_memory_size, random_state=random_state, only_full_history=self._only_full_history)

        self._all_losses = {}

    def _populate_initial_fixed_actions(self, actions):
        for a in actions:
            obs = self._environment.observe()
            reward = self._environment.act(a)
            is_terminal = self._environment.inTerminalState()  # If the transition ends up in a terminal state, mark transition as terminal
            self._addSample(obs, a, reward, is_terminal)


    def _chooseAction(self):

        if self._mode != -1:
            # Act according to the test policy if not in training mode
            action, V = self._test_policy.action(self._state, mode=self._mode, dataset=self._dataset)
        else:
            if self._dataset.n_elems >= self._replay_start_size and not self.gather_data:
                # follow the train policy

                # check if we can use obs_per_state as batch dimension
                state = copy.deepcopy(self._state)
                for i in range(len(self._state)):
                    if state[i].shape[0] > 1:
                        state[i] = np.expand_dims(state[i], axis=0)
                action, V = self._train_policy.action(state, mode=None,
                                                      dataset=self._dataset)
            else:
                # Still gathering initial data: choose dummy action
                action, V = self._train_policy.randomAction()

        for c in self._controllers: c.onActionChosen(self, action)
        return action, V

    def _validate(self):
        states, actions, rewards, next_states, terminals, rndIndices = \
            self._valid_dataset.randomBatch(self._valid_dataset.n_elems, self._exp_priority)

        repr_losses = self._learning_algo.train_repr(
            states[0], actions, rewards, next_states[0],
            terminals.astype(float), training=False)

        return sum(v for v in repr_losses.values())

    def _append_losses(self, repr_losses, transition_loss_ind, n=1000):
        for key, value in repr_losses.items():
            if key not in self._all_losses.keys():
                self._all_losses[key] = []

            if len(self._all_losses[key]) > 2 * n:
                self._all_losses[key] = self._all_losses[key][1:]
            self._all_losses[key].append(value)

        if 'total_loss' not in self._all_losses:
            self._all_losses['total_loss'] = []
        if len(self._all_losses['total_loss']) > 2 * n:
            self._all_losses['total_loss'] = self._all_losses['total_loss'][1:]
        self._all_losses['total_loss'].append(sum(v for v in repr_losses.values()))

        if 'inference_transition_loss' not in self._all_losses:
            self._all_losses['inference_transition_loss'] = []

        if len(self._all_losses['inference_transition_loss']) > 2 * n:
            self._all_losses['inference_transition_loss'] = self._all_losses['inference_transition_loss'][1:]

        self._all_losses['inference_transition_loss'].append(np.average(transition_loss_ind))


    def train(self):
        if self._dataset.n_elems < self._replay_start_size:
            if not self._populated and self._add_single_cycle:
                self._populate_initial_fixed_actions(self._initial_actions)
                self._populated = True
            return

        count = 0
        n = 2000
        cont_cond = True

        iters_to_run = self._iters_per_update
        if self._num_train_steps < 4 and self._learn_representation:
            iters_to_run = self._iters_per_update * (4 - self._num_train_steps)

        # import tracemalloc
        # from deer.helper.mem_profiler import display_top
        #
        # tracemalloc.start()

        while cont_cond:

            cont_cond = count < iters_to_run
            # randomly sample from dataset n times
            batch = self._dataset.randomBatch_nstep(self._batch_size, self._train_nstep, self._exp_priority)
            nstep_states = batch['observations']
            nstep_actions = batch['actions']
            nstep_rewards = batch['rewards']
            nstep_terminals = batch['terminals']
            rndIndices = batch['rndValidIndices']
            nstep_masks = None
            if 'masks' in batch:
                nstep_masks = batch['masks']

            if self._secondary_rewards:
                nstep_secondary_rewards = batch['secondary_rewards']
                # We can do most specifying for reward learning here
                # MIGHT need to change in the future
                if self._reward_learning == "only_secondary":
                    nstep_rewards = nstep_secondary_rewards

            # TEST for secondary rewards here
            if self._learn_representation:
                repr_losses, random_states_loss_ind, transition_loss_ind = \
                    self._learning_algo.train_repr(nstep_states[0], nstep_actions, nstep_rewards, nstep_terminals.astype(float), scale=self._dataset.n_elems)

                if 'trans_rand_ent_combined' not in self._all_losses:
                    self._all_losses['trans_rand_ent_combined'] = []
                if 'two_random_state_entropy_max_loss' in repr_losses:
                    if len(self._all_losses['trans_rand_ent_combined']) > n * 10:
                        self._all_losses['trans_rand_ent_combined'] = self._all_losses['trans_rand_ent_combined'][1:]
                    self._all_losses['trans_rand_ent_combined'].append(repr_losses['transition_loss'] + repr_losses['two_random_state_entropy_max_loss'])

                self._append_losses(repr_losses, transition_loss_ind, n=n)

                if count % n == 0 and count > 0:
                    last_n_transition_losses = self._all_losses['inference_transition_loss'][-(n + 1):]
                    avg_transition_loss = np.average(last_n_transition_losses)
                    cont_cond = (avg_transition_loss >
                                 (self._consec_dist / self._slack_ratio)**2)

                    # last_n_normalized_trans_losses = self._all_losses['inference_normalized_transition_losses'][-(n + 1):]
                    # avg = np.average(last_n_normalized_trans_losses)
                    # cont_cond = avg < 0.25

                if (self._exp_priority):
                    loss_ind = random_states_loss_ind + transition_loss_ind
                    self._dataset.updatePriorities(pow(loss_ind,self._exp_priority)+0.0001, rndIndices[1])

            if count > 0 and self._train_q and \
                    (self._num_train_steps > 0 or not self._learn_representation):
                # TEST for secondary rewards here
                if self._secondary_rewards and self._reward_learning == "combined":
                    # This ensures that we learn a Q value for both intr and extr rewards
                    nstep_rewards += nstep_secondary_rewards

                td_err, inds = self._learning_algo.train_q(
                    nstep_states[0], nstep_actions, nstep_rewards, nstep_terminals, nstep_masks=nstep_masks)
                if 'td_err' not in self._all_losses.keys():
                    self._all_losses['td_err'] = []

                self._all_losses['td_err'].append(td_err.item())

            # if count % 500 == 0:
            # snapshot = tracemalloc.take_snapshot()
            # display_top(snapshot)
            # print(f"size of _all_losses: {asizeof(self._all_losses)}")
            # print(f"size of dataset: {asizeof(self._dataset)}")
            # print("here")

            for c in self._controllers: c.onTrainStepTaken(self)

            count += 1

        for c in self._controllers: c.onTrainLoopTaken(self)

        self._num_train_steps += 1



import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

from nsrl.helper.data import CircularBuffer, SliceError
from nsrl.helper import tree

try:
    import cPickle as pickle
except ModuleNotFoundError:
    import pickle

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

def calculate_large_batch(model, data, smaller_bsize=32):
    """
    Used to calculate a large batch. We're batching our large batch into smaller batches.
    batch.
    ONLY WORKS IN INFERENCE.
    :param model: model needed to pass data through
    :param data:
    :param smaller_bsize: smaller batch size.
    :return: combined
    """
    with torch.no_grad():
        batches = data.shape[0] // smaller_bsize
        results = []
        batches = batches + 2 if batches * smaller_bsize < data.shape[0] else batches + 1
        for i in range(1, batches):
            batch = data[(i - 1) * smaller_bsize:i * smaller_bsize]
            batch = torch.tensor(batch, dtype=torch.float).to(device)
            result = model(batch)
            results.append(result)
        return torch.cat(results, dim=0)



class DataSet(Dataset):
    """
    Pytorh implementation of Replay memory. Allows iteration through entire dataset.
    Prioritized Experience Replay doesn't work for this DataSet!
    """

    def __init__(self, env, random_state=None, max_size=1000000, use_priority=False, only_full_history=True):
        super(DataSet, self).__init__()
        self._environment = env
        self._batch_dimensions = env.inputDimensions()
        self._max_history_size = np.max([self._batch_dimensions[i][0] for i in range (len(self._batch_dimensions))])
        self._size = max_size
        self._use_priority = use_priority
        self._only_full_history = only_full_history

        if use_priority:
            raise Exception("Priority replay not available for PyTorch Dataset.")

        if ( isinstance(env.nActions(),int) ):
            self._actions      = CircularBuffer(max_size, dtype="int8")
        else:
            self._actions      = CircularBuffer(max_size, dtype='object')
        self._rewards      = CircularBuffer(max_size)
        self._terminals    = CircularBuffer(max_size, dtype="bool")
        if (self._use_priority):
            self._prioritiy_tree = tree.SumTree(max_size)
            self._translation_array = np.zeros(max_size)

        self._observations = np.zeros(len(self._batch_dimensions), dtype='object')
        # Initialize the observations container if necessary
        for i in range(len(self._batch_dimensions)):
            self._observations[i] = CircularBuffer(max_size, elemShape=self._batch_dimensions[i][1:], dtype=env.observationType(i))

        if (random_state == None):
            self._random_state = np.random.RandomState()
        else:
            self._random_state = random_state

        self.n_elems  = 0
        self.sticky_action=1        # Number of times the agent is forced to take the same action as part of one actual time step

    def __len__(self):
        return self.n_elems

    def __getitem__(self, idx):

        if (self._max_history_size + self.sticky_action - 1 >= self.n_elems):
            raise SliceError(
                "Not enough elements in the dataset to create a "
                "complete state. {} elements in dataset; requires {}"
                    .format(self.n_elems, self._max_history_size))

        actions = self._actions.getSlice(idx, idx + 1)
        rewards = self._rewards.getSlice(idx, idx + 1)
        terminals = self._terminals.getSlice(idx, idx + 1)

        states = []

        for i in range(len(self._batch_dimensions)):
            state_w_next_state = [self._observations[i].getSlice(idx, idx + 1)[0]]

            if idx == self._observations[i].getIndex():
                state_w_next_state.append(self._environment.observe()[i])
            elif idx == self._observations[i].getTrueSize():
                # If we're not at the last obs of the buffer but we're at the last
                # index of the buffer, go around to the beginning.
                state_w_next_state.append(self._observations[i].getSlice(0, 1)[0])
            else:
                state_w_next_state.append(self._observations[i].getSlice(idx + 1, idx + 2)[0])
            state_w_next_state = np.array(state_w_next_state)
            states.append(state_w_next_state)

        return states, actions, rewards, terminals, idx


    def save(self, fname):
        """
        Saves the DataSet to specified filename
        :return:
        """
        with open(fname, 'wb') as output:
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(fname):
        with open(fname, 'rb') as inp:
            return pickle.load(inp)

    def addSample(self, obs, action, reward, is_terminal, priority):
        # Store observations
        for i in range(len(self._batch_dimensions)):
            self._observations[i].append(obs[i])

        # Update tree and translation table
        if (self._use_priority):
            index = self._actions.getIndex()
            if (index >= self._size):
                ub = self._actions.getUpperBound()
                true_size = self._actions.getTrueSize()
                tree_ind = index%self._size
                if (ub == true_size):
                    size_extension = true_size - self._size
                    # New index
                    index = self._size - 1
                    tree_ind = -1
                    # Shift translation array
                    self._translation_array -= size_extension + 1
                tree_ind = np.where(self._translation_array==tree_ind)[0][0]
            else:
                tree_ind = index

            self._prioritiy_tree.update(tree_ind)
            self._translation_array[tree_ind] = index

        # Store rest of sample
        self._actions.append(action)
        self._rewards.append(reward)
        self._terminals.append(is_terminal)

        if (self.n_elems < self._size):
            self.n_elems += 1

    def actions(self):
        """Get all actions currently in the replay memory, ordered by time where they were taken."""

        return self._actions.getSlice(0)

    def rewards(self):
        """Get all rewards currently in the replay memory, ordered by time where they were received."""

        return self._rewards.getSlice(0)

    def terminals(self):
        """Get all terminals currently in the replay memory, ordered by time where they were observed.

        terminals[i] is True if actions()[i] lead to a terminal state (i.e. corresponded to a terminal
        transition), and False otherwise.
        """

        return self._terminals.getSlice(0)

    def observations(self):
        """Get all observations currently in the replay memory, ordered by time where they were observed.
        """

        ret = np.zeros_like(self._observations)
        for input in range(len(self._observations)):
            ret[input] = self._observations[input].getSlice(0)

        return ret

    def updateRewards(self, new_rewards, indices):
        self._rewards[indices] = new_rewards

    def updatePriorities(self, priorities, rndValidIndices):
        """
        """
        for i in range( len(rndValidIndices) ):
            self._prioritiy_tree.update(rndValidIndices[i], priorities[i])


""" 
This module contains classes used to define the standard behavior of the agent.
It relies on the controllers, the chosen training/test policy and the learning algorithm
to specify its behavior in the environment.

"""

import numpy as np
import joblib
from warnings import warn

# import tracemalloc

from nsrl.experiment import base_controllers as controllers
from nsrl.policies import EpsilonGreedyPolicy
from nsrl.helper.data import DataSet, SliceError, CircularBuffer

class NeuralAgent(object):
    """The NeuralAgent class wraps a learning algorithm (such as a deep Q-network) for training and testing in a given environment.
    
    Attach controllers to it in order to conduct an experiment (when to train the agent, when to test,...).
    
    Parameters
    -----------
    environment : object from class Environment
        The environment in which the agent interacts
    learning_algo : object from class LearningAlgo
        The learning algorithm associated to the agent
    replay_memory_size : int
        Size of the replay memory. Default : 1000000
    replay_start_size : int
        Number of observations (=number of time steps taken) in the replay memory before starting learning. 
        Default: minimum possible according to environment.inputDimensions().
    batch_size : int
        Number of tuples taken into account for each iteration of gradient descent. Default : 32
    random_state : numpy random number generator
        Default : random seed.
    exp_priority : float
        The exponent that determines how much prioritization is used, default is 0 (uniform priority).
        One may check out Schaul et al. (2016) - Prioritized Experience Replay.
    train_policy : object from class Policy
        Policy followed when in training mode (mode -1)
    test_policy : object from class Policy
        Policy followed when in other modes than training (validation and test modes)
    only_full_history : boolean
        Whether we wish to train the neural network only on full histories or we wish to fill with zeroes the 
        observations before the beginning of the episode
    use_dataloader : boolean
        Whether or not we use the PyTorch DataLoader to iterate through replay.
    """

    def __init__(self, environment,
                 learning_algo, replay_memory_size=1000000,
                 replay_start_size=None, batch_size=32,
                 random_state=np.random.RandomState(),
                 exp_priority=0, train_policy=None,
                 test_policy=None, only_full_history=True,
                 reset_dataset_per_epoch=False,
                 network_fname=None, dataset_fname=None,
                 secondary_rewards=False, dataset=None,
                 reload=False, **kwargs):
        inputDims = environment.inputDimensions()


        if replay_start_size == None:
            replay_start_size = max(inputDims[i][0] for i in range(len(inputDims)))
        elif replay_start_size < max(inputDims[i][0] for i in range(len(inputDims))) :
            raise AgentError("Replay_start_size should be greater than the biggest history of a state.")
        
        self._controllers = []
        self._environment = environment
        self._learning_algo = learning_algo
        self._replay_memory_size = replay_memory_size
        self._replay_start_size = replay_start_size
        self._batch_size = batch_size
        self._random_state = random_state
        self._exp_priority = exp_priority
        self._reload = reload
        self._only_full_history = only_full_history

        # whether or not to use bootstrap Q.
        self._bootstrap_q = kwargs.get('action_type') == 'bootstrap_q'
        self._sample_head_every = kwargs.get('sample_head_every', 20)
        self._bernoulli_p = kwargs.get('bernoulli_p', 0.5)
        head_num = 1 if not hasattr(train_policy, 'head_num') else train_policy.head_num

        self._dataset = dataset if dataset is not None else \
            DataSet(environment, max_size=replay_memory_size,
                    random_state=random_state, use_priority=self._exp_priority,
                    only_full_history=self._only_full_history,
                    secondary_rewards=secondary_rewards, head_num=head_num)
        self._secondary_rewards = secondary_rewards
        self._tmp_dataset = None # Will be created by startTesting() when necessary
        # Validation dataset for validation within training
        # self._valid_dataset = DataSet(environment, max_size=replay_memory_size,
        #                               random_state=random_state, use_priority=self._exp_priority,
        #                               only_full_history=self._only_full_history)
        self._mode = -1
        self._mode_epochs_length = 0
        self._total_mode_reward = 0
        self._training_loss_averages = []
        self._Vs_on_last_episode = []
        self._in_episode = False
        self._selected_action = -1
        self._reset_data_per_epoch = reset_dataset_per_epoch
        self._state = []



        if network_fname is not None:
            print("Setting model and optimizer params")
            self.setNetwork(network_fname)

        if dataset_fname is not None:
            print("Reloading dataset")
            self._dataset = DataSet.load(dataset_fname)
            # Now we need to set the environment state based on dataset.
            # Currently ONLY WORKING for simple_maze_env.
            # This currently implies that environment was saved by dataset. THIS MIGHT NOT BE
            # THE CASE LATER ON FOR NON-SIMPLE_MAZE_ENV ENVIRONMENTS.
            self._environment.loadState(self._dataset, self._dataset._environment)
            self._dataset._environment = self._environment

            self._reload = True

        for i in range(len(inputDims)):
            self._state.append(np.zeros(inputDims[i], dtype=float))
        if (train_policy==None):
            self._train_policy = EpsilonGreedyPolicy(learning_algo, environment.nActions(), random_state, 0.1)
        else:
            self._train_policy = train_policy
        if (test_policy==None):
            self._test_policy = EpsilonGreedyPolicy(learning_algo, environment.nActions(), random_state, 0.)
        else:
            self._test_policy = test_policy
        self.gathering_data=True    # Whether the agent is gathering data or not
        self.sticky_action=1        # Number of times the agent is forced to take the same action as part of one actual time step

    def setControllersActive(self, toDisable, active):
        """ Activate controller
        """
        for i in toDisable:
            self._controllers[i].setActive(active)

    def setLearningRate(self, lr):
        """ Set the learning rate for the gradient descent
        """
        self._learning_algo.setLearningRate(lr)

    def learningRate(self):
        """ Get the learning rate
        """
        return self._learning_algo.learningRate()

    def setDiscountFactor(self, df):
        """ Set the discount factor
        """
        self._learning_algo.setDiscountFactor(df)

    def discountFactor(self):
        """ Get the discount factor
        """
        return self._learning_algo.discountFactor()

    def overrideNextAction(self, action):
        """ Possibility to override the chosen action. This possibility should be used on the signal OnActionChosen.
        """
        self._selected_action = action

    def avgBellmanResidual(self):
        """ Returns the average training loss on the epoch
        """
        if (len(self._training_loss_averages) == 0):
            return -1
        return np.average(self._training_loss_averages)

    def avgEpisodeVValue(self):
        """ Returns the average V value on the episode (on time steps where a non-random action has been taken)
        """
        if (len(self._Vs_on_last_episode) == 0):
            return -1
        if(np.trim_zeros(self._Vs_on_last_episode)!=[]):
            return np.average(np.trim_zeros(self._Vs_on_last_episode))
        else:
            return 0

    def totalRewardOverLastTest(self):
        """ Returns the average sum of rewards per episode and the number of episode
        """
        return self._total_mode_reward/self._totalModeNbrEpisode, self._totalModeNbrEpisode

    def attach(self, controller):
        if (isinstance(controller, controllers.Controller)):
            self._controllers.append(controller)
        else:
            raise TypeError("The object you try to attach is not a Controller.")

    def detach(self, controllerIdx):
        return self._controllers.pop(controllerIdx)

    def mode(self):
        return self._mode

    def startMode(self, mode, epochLength):
        if self._in_episode:
            raise AgentError("Trying to start mode while current episode is not yet finished. This method can be "
                             "called only *between* episodes for testing and validation.")
        elif mode == -1:
            raise AgentError("Mode -1 is reserved and means 'training mode'; use resumeTrainingMode() instead.")
        else:
            self._mode = mode
            self._mode_epochs_length = epochLength
            self._total_mode_reward = 0.
            del self._tmp_dataset
            self._tmp_dataset = DataSet(self._environment, self._random_state,
                                        max_size=self._replay_memory_size, only_full_history=self._only_full_history)

    def resumeTrainingMode(self):
        self._mode = -1

    def summarizeTestPerformance(self):
        if self._mode == -1:
            raise AgentError("Cannot summarize test performance outside test environment.")

        self._environment.summarizePerformance(self._tmp_dataset, self._learning_algo, train_data_set=self._dataset)

    def train(self):
        """
        This function selects a random batch of data (with self._dataset.randomBatch) and performs a 
        Q-learning iteration (with self._learning_algo.train).        
        """
        # We make sure that the number of elements in the replay memory
        # is strictly superior to self._replay_start_size before taking 
        # a random batch and perform training
        if self._dataset.n_elems <= self._replay_start_size:
            return

        try:
            if hasattr(self._learning_algo, 'nstep'):
                batch = self._dataset.randomBatch_nstep(self._batch_size, self._learning_algo.nstep, self._exp_priority)
                observations = batch['observations']
                actions = batch['observations']
                rewards = batch['rewards']
                terminals = batch['terminals']
                rndValidIndices = batch['rndValidIndices']
                loss, loss_ind = self._learning_algo.train(observations, actions, rewards, terminals)
            else:
                batch = self._dataset.randomBatch(self._batch_size, self._exp_priority)
                states = batch['state']
                next_states = batch['next_state']
                actions = batch['observations']
                rewards = batch['rewards']
                terminals = batch['terminals']
                rndValidIndices = batch['rndValidIndices']
                loss, loss_ind = self._learning_algo.train(states, actions, rewards, next_states, terminals)

            # slight redundancy when we have 1 train step in 1 train() call
            for c in self._controllers: c.onTrainStepTaken(self)

            self._training_loss_averages.append(loss)
            if (self._exp_priority):
                self._dataset.updatePriorities(pow(loss_ind,self._exp_priority)+0.0001, rndValidIndices[1])

            for c in self._controllers: c.onTrainLoopTaken(self)

        except SliceError as e:
            warn("Training not done - " + str(e), AgentWarning)

    def dumpNetwork(self, fname, nEpoch=-1):
        """ Dump the network

        Parameters
        -----------
        fname : string
            Name of the file where the network will be dumped
        nEpoch : int
            Epoch number (Optional)
        """
        # try:
        #     os.mkdir("nnets")
        # except Exception:
        #     pass
        # basename = "nnets/" + fname

        # for f in os.listdir("nnets/"):
        #     if fname in f:
        #         os.remove("nnets/" + f)

        all_params = self._learning_algo.getAllParams()
        print("Dumping network to %s" % fname)
        if (nEpoch>=0):
            joblib.dump(all_params, fname + ".epoch={}".format(nEpoch))
        else:
            joblib.dump(all_params, fname, compress=True)

    def setNetwork(self, fname, nEpoch=-1):
        """ Set values into the network

        Parameters
        -----------
        fname : string
            Name of the file where the values are
        nEpoch : int
            Epoch number (Optional)
        """
        basename = fname
        # basename = "nnets/" + fname

        if (nEpoch>=0):
            all_params = joblib.load(basename + ".epoch={}".format(nEpoch))
        else:
            # import torch
            # all_params = torch.load(basename, map_location='cpu')
            all_params = joblib.load(basename)

        self._learning_algo.setAllParams(all_params)

    def runSingleEpisode(self, episodes=1, max_length=1000, preset_actions=None):
        i = 0
        # Set max length here
        if max_length != 1000:
            self._environment.setMaxSteps(max_length)

        for c in self._controllers: c.onStart(self)

        while i < episodes:
            self._runEpisode(max_length, preset_actions=preset_actions)
            i += 1

        self._environment.end()
        for c in self._controllers: c.onEnd(self)

    def run(self, n_epochs, epoch_length, break_on_done=False, start_count=0):
        """
        This function encapsulates the whole process of the learning.
        It starts by calling the controllers method "onStart",
        Then it runs a given number of epochs where an epoch is made up of one or many episodes (called with
        agent._runEpisode) and where an epoch ends up after the number of steps reaches the argument "epoch_length".
        It ends up by calling the controllers method "end".

        Parameters
        -----------
        n_epochs : int
            number of epochs
        epoch_length : int
            maximum number of steps for a given epoch
        """
        for c in self._controllers: c.onStart(self)
        i = 0
        while i < n_epochs or self._mode_epochs_length > 0:
            self._training_loss_averages = []

            if self._mode != -1:
                self._totalModeNbrEpisode=0
                while self._mode_epochs_length > 0:
                    self._totalModeNbrEpisode += 1
                    self._mode_epochs_length = self._runEpisode(self._mode_epochs_length - start_count)
                    start_count = 0
            else:
                length = epoch_length
                while length > 0:
                    length = self._runEpisode(length, break_on_done=break_on_done)
                    if break_on_done:
                        break
                i += 1
            for c in self._controllers: c.onEpochEnd(self)

        self._environment.end()
        for c in self._controllers: c.onEnd(self)

    def _runEpisode(self, maxSteps, preset_actions=None, break_on_done=False):
        """
        This function runs an episode of learning. An episode ends up when the environment method "inTerminalState"
        returns True (or when the number of steps reaches the argument "maxSteps")

        Parameters
        -----------
        maxSteps : int
            maximum number of steps before automatically ending the episode
        """
        self._in_episode = True

        if not self._reload:
            self._environment.reset(self._mode)

        initState = self._environment.observe()

        if self._reset_data_per_epoch:
            self._dataset = DataSet(self._environment, max_size=self._replay_memory_size,
                                    random_state=self._random_state, use_priority=self._exp_priority, only_full_history=self._only_full_history)

        if self._bootstrap_q and hasattr(self._train_policy, 'sample_head'):
            self._train_policy.sample_head()

        inputDims = self._environment.inputDimensions()
        for i in range(len(inputDims)):
            if inputDims[i][0] > 1 and self._state[i][1:].shape == initState[i][1:].shape:
                self._state[i][1:] = initState[i][1:]
            elif inputDims[i][0] > 1:
                self._state[i] = np.repeat(initState[0][None, :, :], inputDims[i][0], axis=0)

        self._Vs_on_last_episode = []
        is_terminal=False
        reward=0
        j = 0
        # Set preset actions here
        while maxSteps > 0:

            maxSteps -= 1
            if(self.gathering_data==True or self._mode!=-1):
                if j > 0 and self._bootstrap_q and j % self._sample_head_every == 0:
                    self._train_policy.sample_head()
                obs = self._environment.observe()

                for i in range(len(obs)):
                    self._state[i][0:-1] = self._state[i][1:]
                    self._state[i][-1] = obs[i]

                action = None
                if preset_actions is not None:
                    action = preset_actions[j % len(preset_actions)]


                V, action, reward, add_steps = self._step(action=action)
                maxSteps -= add_steps

                self._Vs_on_last_episode.append(V)
                if self._mode != -1:
                    self._total_mode_reward += reward

                is_terminal = self._environment.inTerminalState()   # If the transition ends up in a terminal state, mark transition as terminal
                                                                    # Note that the new obs will not be stored, as it is unnecessary.

                if(maxSteps>0):
                    self._addSample(obs, action, reward, is_terminal)
                else:
                    self._addSample(obs, action, reward, True)      # If the episode ends because max number of steps is reached, mark the transition as terminal

                if break_on_done and is_terminal:
                    break
                j += 1
            for c in self._controllers: c.onActionTaken(self)

            if is_terminal:
                # We need this with Monitor, as it
                # needs to exactly run n steps, and no more.
                # maxSteps = 0
                break

        self._in_episode = False
        for c in self._controllers: c.onEpisodeEnd(self, is_terminal, reward)
        return maxSteps


    def _step(self, action=None, V=0):
        """
        This method is called at each time step and performs one action in the environment.

        Returns
        -------
        V : float
            Estimated value function of current state.
        action : int
            The id of the action selected by the agent.
        reward : float
            Reward obtained for the transition
        """
        if action is None:
            action, V = self._chooseAction()
        reward=0
        for i in range(self.sticky_action):
            reward += self._environment.act(action)

        return V, action, reward, self.sticky_action - 1

    def _addSample(self, ponctualObs, action, reward, is_terminal):
        mask = None
        if self._bootstrap_q:
            head_num = self._train_policy.head_num
            mask = np.random.binomial(n=1, p=self._bernoulli_p, size=head_num)

        if self._mode != -1:
            self._tmp_dataset.addSample(ponctualObs, action, reward, is_terminal, priority=1, mask=mask)
        else:
            self._dataset.addSample(ponctualObs, action, reward, is_terminal, priority=1, mask=mask)


    def _chooseAction(self):

        if self._mode != -1:
            # Act according to the test policy if not in training mode
            action, V = self._test_policy.action(self._state, mode=self._mode, dataset=self._dataset)
        else:
            if self._dataset.n_elems > self._replay_start_size:
                # follow the train polic4
                # potentially add action_type here for planning
                action, V = self._train_policy.action(self._state, mode=None, dataset=self._dataset)     #is self._state the only way to store/pass the state?
            else:
                # Still gathering initial data: choose dummy action
                action, V = self._train_policy.randomAction()

        for c in self._controllers: c.onActionChosen(self, action)
        return action, V

class AgentError(RuntimeError):
    """Exception raised for errors when calling the various Agent methods at wrong times.
    Attributes:
        expr -- input expression in which the error occurred
        msg  -- explanation of the error
    """

    def __init__(self, value):
        self.value = value
    def __str__(self):
        return repr(self.value)

class AgentWarning(RuntimeWarning):
    """Warning issued of the various Agent methods.
    Attributes:
        expr -- input expression in which the error occurred
        msg  -- explanation of the error
    """




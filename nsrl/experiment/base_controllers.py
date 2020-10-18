import numpy as np
import joblib
import os

from nsrl.helper.video import add_label_to_frames, frames_to_video
from nsrl.helper.data import DataSet

class Controller(object):
    """A base controller that does nothing when receiving the various signals emitted by an agent. This class should 
    be the base class of any controller you would want to define.
    """

    def __init__(self, **kwargs):
        """Activate this controller.

        All controllers inheriting this class should call this method in their own __init()__ using 
        super(self.__class__, self).__init__().
        """

        self._active = True
        self._count = kwargs.get('start_count', 0)

    def setActive(self, active):
        """Activate or deactivate this controller.
        
        A controller should not react to any signal it receives as long as it is deactivated. For instance, if a 
        controller maintains a counter on how many episodes it has seen, this counter should not be updated when 
        this controller is disabled.
        """

        self._active = active

    def onStart(self, agent):
        """Called when the agent is going to start working (before anything else).
        
        This corresponds to the moment where the agent's run() method is called.

        Parameters
        ----------
             agent : NeuralAgent
                The agent firing the event
        """

        pass

    def onEpisodeEnd(self, agent, terminal_reached, reward):
        """Called whenever the agent ends an episode, just after this episode ended and before any onEpochEnd() signal
        could be sent.

        Parameters
        ----------
        agent : NeuralAgent
            The agent firing the event
        terminal_reached : bool
            Whether the episode ended because a terminal transition occured. This could be False 
            if the episode was stopped because its step budget was exhausted.
        reward : float
            The reward obtained on the last transition performed in this episode.
        
        """

        pass

    def onEpochEnd(self, agent):
        """Called whenever the agent ends an epoch, just after the last episode of this epoch was ended and after any 
        onEpisodeEnd() signal was processed.

        Parameters
        ----------
        agent : NeuralAgent
            The agent firing the event
        """

        pass

    def onActionChosen(self, agent, action):
        """Called whenever the agent has chosen an action.

        This occurs after the agent state was updated with the new observation it made, but before it applied this 
        action on the environment and before the total reward is updated.
        """

        pass

    def onActionTaken(self, agent):
        """Called whenever the agent has taken an action on its environment.

        This occurs after the agent applied this action on the environment and before terminality is evaluated. This 
        is called only once, even in the case where the agent skip frames by taking the same action multiple times.
        In other words, this occurs just before the next observation of the environment.
        """

        pass

    def onTrainStepTaken(self, agent):
        """
        Called whenever the agent finishes a single training step.
        Multiple training steps aggregate to a training loop.
        """
        pass

    def onTrainLoopTaken(self, agent):
        """
        Called after .train() call is finished.
        """
        pass

    def onEnd(self, agent):
        """Called when the agent has finished processing all its epochs, just before returning from its run() method.
        """

        pass


class LearningRateController(Controller):
    """A controller that modifies the learning rate periodically upon epochs end.
    
    Parameters
    ----------
    initial_learning_rate : float
        The learning rate upon agent start
    learning_rate_decay : float
        The factor by which the previous learning rate is multiplied every [periodicity] epochs.
    periodicity : int
        How many epochs are necessary before an update of the learning rate occurs
    """

    def __init__(self, initial_learning_rate=0.005, learning_rate_decay=1., periodicity=1):
        """Initializer.

        """
        super(self.__class__, self).__init__()
        self._epoch_count = 0
        self._init_lr = initial_learning_rate
        self._lr = initial_learning_rate
        self._lr_decay = learning_rate_decay
        self._periodicity = periodicity

    def onStart(self, agent):
        if (self._active == False):
            return

        self._epoch_count = 0
        agent._learning_algo.setLearningRate(self._init_lr)
        self._lr = self._init_lr * self._lr_decay

    def onEpochEnd(self, agent):
        if (self._active == False):
            return

        self._epoch_count += 1
        if self._periodicity <= 1 or self._epoch_count % self._periodicity == 0:
            agent._learning_algo.setLearningRate(self._lr)
            self._lr *= self._lr_decay

class EpsilonController(Controller):
    """ A controller that modifies the probability "epsilon" of taking a random action periodically.
    
    Parameters
    ----------
    initial_e : float
        Start epsilon
    e_decays : int
        How many updates are necessary for epsilon to reach eMin
    e_min : float
        End epsilon
    evaluate_on : str
        After what type of event epsilon shoud be updated periodically. Possible values: 'action', 'episode', 'epoch'.
    periodicity : int
        How many [evaluateOn] are necessary before an update of epsilon occurs
    reset_every : str
        After what type of event epsilon should be reset to its initial value. Possible values: 
        'none', 'episode', 'epoch'.
    """

    def __init__(self, initial_e=1., e_decays=10000, e_min=0.1, evaluate_on='action', periodicity=1, reset_every='none'):
        """Initializer.
        """

        super(self.__class__, self).__init__()
        self._count = 0
        self._init_e = initial_e
        self._e = initial_e
        self._e_min = e_min
        self._e_decay = (initial_e - e_min) / e_decays
        self._periodicity = periodicity

        self._on_action = 'action' == evaluate_on
        self._on_episode = 'episode' == evaluate_on
        self._on_epoch = 'epoch' == evaluate_on
        if not self._on_action and not self._on_episode and not self._on_epoch:
            self._on_action = True

        self._reset_on_episode = 'episode' == reset_every
        self._reset_on_epoch = 'epoch' == reset_every

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

    def onActionChosen(self, agent, action):
        if (self._active == False):
            return

        if self._on_action:
            self._update(agent)

    def _reset(self, agent):
        self._count = 0
        agent._train_policy.setEpsilon(self._init_e)
        self._e = self._init_e

    def _update(self, agent):
        self._count += 1
        if self._periodicity <= 1 or self._count % self._periodicity == 0:
            agent._train_policy.setEpsilon(self._e)
            self._e = max(self._e - self._e_decay, self._e_min)



class DiscountFactorController(Controller):
    """A controller that modifies the q-network discount periodically.
    More informations in : Francois-Lavet Vincent et al. (2015) - How to Discount Deep Reinforcement Learning: Towards New Dynamic Strategies (http://arxiv.org/abs/1512.02011).

    Parameters
    ----------
    initial_discount_factor : float
        Start discount
    discount_factor_growth : float
        The factor by which the previous discount is multiplied every [periodicity]
        epochs.
    discount_factor_max : float
        Maximum reachable discount
    periodicity : int
        How many training epochs are necessary before an update of the discount occurs
    """
    
    def __init__(self, initial_discount_factor=0.9,
                 discount_factor_growth=1., discount_factor_max=0.99,
                 periodicity=1, evaluate_on='epoch'):
        """Initializer.
        """

        super(self.__class__, self).__init__()
        self._count = 0
        self._init_df = initial_discount_factor
        self._df = initial_discount_factor
        self._df_growth = discount_factor_growth
        self._df_max = discount_factor_max
        self._periodicity = periodicity

        self._on_action = 'action' == evaluate_on
        self._on_episode = 'episode' == evaluate_on
        self._on_epoch = 'epoch' == evaluate_on

    def onStart(self, agent):
        if (self._active == False):
            return

        self._count = 0
        agent._learning_algo.setDiscountFactor(self._init_df)
        if (self._init_df < self._df_max):
            self._df = 1 - (1 - self._init_df) * self._df_growth
        else:
            self._df = self._init_df

    def onEpochEnd(self, agent):
        if (self._active == False):
            return

        if self._on_epoch:
            self._update(agent)

    def onActionChosen(self, agent, action):
        if (self._active == False):
            return

        if self._on_action:
            self._update(agent)

    def _update(self, agent):
        self._count += 1
        if self._periodicity <= 1 or self._count % self._periodicity == 0:
            if self._df < self._df_max:
                agent._learning_algo.setDiscountFactor(self._df)
                self._df = 1 - (1 - self._df) * self._df_growth


class TestRepresentationController(Controller):
    """
    Test representation controller. We essentially test on a new environment and get
    video/representations to see if things are working as they should be.
    """
    def __init__(self, test_env, plotter, video_dir, evaluate_on='action', episode_length=500, periodicity=2, summarize_every=10):
        super(TestRepresentationController, self).__init__()

        self.env = test_env
        self._dataset = DataSet(self.env, max_size=episode_length)
        self._plotter = plotter

        self._on_train_loop = 'train_loop' == evaluate_on
        self._on_train_step = 'train_step' == evaluate_on
        self._before_action = 'before_action' == evaluate_on
        self._on_action = 'action' == evaluate_on
        self._on_episode = 'episode' == evaluate_on
        self._on_epoch = 'epoch' == evaluate_on

        self._video_dir = video_dir
        self._periodicity = periodicity
        self._episode_length = episode_length
        self._summarize_every = summarize_every
        self._count = 0


    def onActionTaken(self, agent):
        if self._on_action and self._count > 0:
            if self._count % self._periodicity == 0:
                self._update(agent)
            self._count += 1


    def onActionChosen(self, agent, action):
        if self._before_action and self._count > 0:
            if self._count % self._periodicity == 0:
                self._update(agent)
            self._count += 1

    def onEpochEnd(self, agent):
        if self._on_epoch and self._count > 0:
            if self._count % self._periodicity == 0:
                self._update(agent)
            self._count += 1


    def onTrainLoopTaken(self, agent):
        if self._on_train_loop and self._count > 0:
            if self._count % self._periodicity == 0:
                self._update(agent)
            self._count += 1

    def onTrainStepTaken(self, agent):
        if self._on_train_step and self._count > 0:
            if self._count % self._periodicity == 0:
                self._update(agent)
            self._count += 1

    def _update(self, agent):
        print("Starting testing episode")
        self.env.reset(0)
        steps = 0
        state = self.env.observe()[0]
        frames = [add_label_to_frames(state, steps)]
        terminal = self.env.inTerminalState()
        while not terminal and steps < self._episode_length:
            print(f"Testing step {steps}")
            action, v = agent._test_policy.action([np.expand_dims(state, axis=0)], mode=0, dataset=agent._dataset)
            reward = self.env.act(action)
            terminal = self.env.inTerminalState()
            self._dataset.addSample(state, action, reward, terminal, priority=1)
            steps += 1
            state = self.env.observe()[0]
            frames.append(add_label_to_frames(state, steps))

        all_obs = np.stack(frames)
        all_obs_flattened = np.concatenate(all_obs, axis=0)
        path = os.path.join(self._video_dir, f'test_episode_{self._count}.mp4')

        frames_to_video(all_obs_flattened, path, scale=3.0)

        fig = self.env.summarizePerformance(self._dataset, agent._learning_algo)

        self._plotter.plot_plotly_fig(f'test_abstr_{self._count}', fig,
                                      title_name=f'test_abstr_{self._count}')
        print(f"Test representation done for step {self._count}")



class InterleavedTestEpochController(Controller):
    """A controller that interleaves a test epoch between training epochs of the agent.
    
    Parameters
    ----------
    id : int
        The identifier (>= 0) of the mode each test epoch triggered by this controller will belong to. 
        Can be used to discriminate between datasets in your Environment subclass (this is the argument that 
        will be given to your environment's reset() method when starting the test epoch).
    epoch_length : float
        The total number of transitions that will occur during a test epoch. This means that
        this epoch could feature several episodes if a terminal transition is reached before this budget is 
        exhausted.
    controllers_to_disable : list of int
        A list of controllers to disable when this controller wants to start a
        test epoch. These same controllers will be reactivated after this controller has finished dealing with
        its test epoch.
    periodicity : int 
        How many epochs are necessary before a test epoch is ran (these controller's epochs
        included: "1 test epoch on [periodicity] epochs"). Minimum value: 2.
    show_score : bool
        Whether to print an informative message on stdout at the end of each test epoch, about 
        the total reward obtained in the course of the test epoch.
    summarize_every : int
        How many of this controller's test epochs are necessary before the attached agent's 
        summarizeTestPerformance() method is called. Give a value <= 0 for "never". If > 0, the first call will
        occur just after the first test epoch.
    """

    def __init__(self, id=0, epoch_length=500, controllers_to_disable=[], periodicity=2, show_score=True, summarize_every=10):
        """Initializer.
        """

        super(self.__class__, self).__init__()
        self._epoch_count = 0
        self._id = id
        self._epoch_length = epoch_length
        self._to_disable = controllers_to_disable
        self._show_score = show_score
        if periodicity <= 2:
            self._periodicity = 2
        else:
            self._periodicity = periodicity

        self._summary_counter = 0
        self._summary_periodicity = summarize_every

    def onStart(self, agent):
        if (self._active == False):
            return

        self._epoch_count = 0
        self._summary_counter = 0

    def onEpochEnd(self, agent):
        if (self._active == False):
            return

        mod = self._epoch_count % self._periodicity
        self._epoch_count += 1
        if mod == 0:
            agent.startMode(self._id, self._epoch_length)
            agent.setControllersActive(self._to_disable, False)
        elif mod == 1:
            self._summary_counter += 1
            if self._show_score:
                score,nbr_episodes=agent.totalRewardOverLastTest()
                print("Testing score per episode (id: {}) is {} (average over {} episode(s))".format(self._id, score, nbr_episodes))
            if self._summary_periodicity > 0 and self._summary_counter % self._summary_periodicity == 0:
                agent.summarizeTestPerformance()
            agent.resumeTrainingMode()
            agent.setControllersActive(self._to_disable, True)


class TrainerController(Controller):
    """A controller that makes the agent train on its current database periodically.

    Parameters
    ----------
    evaluate_on : str
        After what type of event the agent shoud be trained periodically. Possible values: 
        'action', 'episode', 'epoch'. The first training will occur after the first occurence of [evaluateOn].
    periodicity : int
        How many [evaluateOn] are necessary before a training occurs
        _show_avg_Bellman_residual [bool] - Whether to show an informative message after each episode end (and after a 
        training if [evaluateOn] is 'episode') about the average bellman residual of this episode
    show_episode_avg_V_value : bool
        Whether to show an informative message after each episode end (and after a 
        training if [evaluateOn] is 'episode') about the average V value of this episode
    """
    def __init__(self, evaluate_on='action', periodicity=1, show_episode_avg_V_value=True, show_avg_Bellman_residual=True):
        """Initializer.
        """

        super(self.__class__, self).__init__()
        self._count = 0
        self._periodicity = periodicity
        self._show_avg_Bellman_residual = show_avg_Bellman_residual
        self._show_episode_avg_V_value = show_episode_avg_V_value

        self._on_action = 'action' == evaluate_on
        self._on_episode = 'episode' == evaluate_on
        self._on_epoch = 'epoch' == evaluate_on
        if not self._on_action and not self._on_episode and not self._on_epoch:
            self._on_action = True

    def onStart(self, agent):
        if (self._active == False):
            return
        
        self._count = 0

    def onEpisodeEnd(self, agent, terminal_reached, reward):
        if (self._active == False):
            return
        
        if self._on_episode:
            self._update(agent)

        if self._show_avg_Bellman_residual: print("Average (on the epoch) training loss: {}".format(agent.avgBellmanResidual()))
        if self._show_episode_avg_V_value: print("Episode average V value: {}".format(agent.avgEpisodeVValue())) # (on non-random action time-steps)

    def onEpochEnd(self, agent):
        if (self._active == False):
            return

        if self._on_epoch:
            self._update(agent)

    def onActionTaken(self, agent):
        if (self._active == False):
            return

        if self._on_action:
            self._update(agent)

    def _update(self, agent):
        if self._periodicity <= 1 or self._count % self._periodicity == 0:
            agent.train()
        self._count += 1
            

class VerboseController(Controller):
    """A controller that print various agent information periodically:
    
    * Count of passed [evaluateOn]
    * Agent current learning rate
    * Agent current discount factor
    * Agent current epsilon

    Parameters
    ----------
    evaluate_on : str
        After what type of event the printing should occur periodically. Possible values: 
        'action', 'episode', 'epoch'. The first printing will occur after the first occurence of [evaluateOn].
    periodicity : int
        How many [evaluateOn] are necessary before a printing occurs
    """

    def __init__(self, evaluateOn=False, evaluate_on='epoch', periodicity=1, **kwargs):
        """Initializer.
        """
        if evaluateOn is not False:
            raise Exception('For uniformity the attributes to be provided to the controllers respect PEP8 from deer0.3dev1 onwards. For instance, instead of "evaluateOn", you should now have "evaluate_on". Please have a look at https://github.com/VinF/deer/issues/28.')

        super(self.__class__, self).__init__(**kwargs)
        self._periodicity = periodicity
        self._string = evaluate_on

        self._on_action = 'action' == evaluate_on
        self._on_episode = 'episode' == evaluate_on
        self._on_epoch = 'epoch' == evaluate_on
        if not self._on_action and not self._on_episode and not self._on_epoch:
            self._on_epoch = True

    def onStart(self, agent):
        if (self._active == False):
            return
        
        self._count = agent._dataset.n_elems - 1

    def onEpisodeEnd(self, agent, terminal_reached, reward):
        if (self._active == False):
            return
        
        if self._on_episode:
            self._print(agent)

    def onEpochEnd(self, agent):
        if (self._active == False):
            return

        if self._on_epoch:
            self._print(agent)

    def onActionTaken(self, agent):
        if (self._active == False):
            return

        if self._on_action:
            self._print(agent)

    def _print(self, agent):
        if self._periodicity <= 1 or self._count % self._periodicity == 0:
            print("{} {}:".format(self._string, self._count + 1))
            print("Learning rate: {}".format(agent._learning_algo.learningRate()))
            print("Discount factor: {}".format(agent._learning_algo.discountFactor()))
            print("Epsilon: {}".format(agent._train_policy.epsilon()))
        self._count += 1



class VideoRecordingController(Controller):
    def __init__(self, evaluate_on='train_loop', periodicity=1,
                 frames_per_video=100, **kwargs):
        super(VideoRecordingController, self).__init__(**kwargs)

        self._on_train_loop = 'train_loop' == evaluate_on
        self._on_train_step = 'train_step' == evaluate_on
        self._before_action = 'before_action' == evaluate_on
        self._on_action = 'action' == evaluate_on
        self._on_episode = 'episode' == evaluate_on
        self._on_epoch = 'epoch' == evaluate_on

        self._periodicity = periodicity
        self._frames_per_video = frames_per_video

        self._skip_first = kwargs.get('skip_first', 0)

    def onActionTaken(self, agent):
        if self._on_action:
            if self._count % self._periodicity == 0 and self._count > self._skip_first:
                self._update(agent)
            self._count += 1

    def onActionChosen(self, agent, action):
        if self._before_action:
            if self._count % self._periodicity == 0 and self._count > self._skip_first:
                self._update(agent)
            self._count += 1

    def onEpochEnd(self, agent):
        if self._on_epoch:
            if self._count % self._periodicity == 0 and self._count > self._skip_first:
                self._update(agent)
            self._count += 1

    def onEpisodeEnd(self, agent, terminal_reached, reward):
        if self._on_episode or terminal_reached:
            self._update(agent)

    def onTrainLoopTaken(self, agent):
        if self._on_train_loop:
            if self._count % self._periodicity == 0 and self._count > 0:
                self._update(agent)
            self._count += 1

    def _update(self, agent):
        # observations = np.expand_dims(observations, axis=-1)
        #
        # agent._plotter.plot_video("video_step_%d" % self._count, observations)
        # print("it worked?")

        agent._environment.env.reset_video_recorder()

class FramesToVideoController(VideoRecordingController):
    def __init__(self, video_dir_path, evaluate_on='train_loop', periodicity=1,
                 frames_per_video=100, **kwargs):
        super(FramesToVideoController, self).__init__(evaluate_on=evaluate_on, periodicity=periodicity,
                                                     frames_per_video=frames_per_video, **kwargs)
        self._video_dir_path = video_dir_path

    def onEpisodeEnd(self, agent, terminal_reached, reward):
        pass

    def onTrainStepTaken(self, agent):
        # For debugging purposes
        pass

    def _update(self, agent):

        # observations = agent._dataset.observations()[0][-(self._frames_per_video - 1):]
        # observations_flat = observations.reshape((-1, observations.shape[-2], observations.shape[-1]))
        # last_observation = agent._environment.observe()[0]
        # observations = np.concatenate((observations_flat, last_observation), axis=0)

        observations = agent._dataset.observationsMatchingBatchDim()[0]
        elems = agent._dataset.n_elems
        # observations = observations[-(self._frames_per_video-1):]
        last_observation = agent._environment.observe()[0]
        last_observation = np.expand_dims(last_observation, axis=0)
        observations = np.concatenate([observations, last_observation], axis=0)

        for i in range(elems - self._frames_per_video, elems):
            observations[i] = add_label_to_frames(observations[i], i)

        flattened_observations = np.concatenate(observations[elems - self._frames_per_video:elems], axis=0)
        video_path = os.path.join(self._video_dir_path, str(self._count) + '_steps.mp4')

        frames_to_video(flattened_observations, video_path, scale=4.0)
        print(f"Converted video for step {self._count}")

class RewardPlottingController(Controller):
    def __init__(self, plotter, evaluate_on='action', periodicity=1,
                 include_secondary_rewards=False, **kwargs):
        super(RewardPlottingController, self).__init__(**kwargs)
        self._on_train_loop = 'train_loop' == evaluate_on
        self._on_train_step = 'train_step' == evaluate_on
        self._before_action = 'before_action' == evaluate_on
        self._on_action = 'action' == evaluate_on
        self._on_episode = 'episode' == evaluate_on
        self._on_epoch = 'epoch' == evaluate_on
        self._evaluate_on = evaluate_on

        self._periodicity = periodicity
        self._plotter = plotter
        self._include_secondary_rewards = include_secondary_rewards

    def onTrainLoopTaken(self, agent):
        if self._on_train_loop:
            if self._count % self._periodicity == 0:
                self._update(agent)
            self._count += 1

    def onActionTaken(self, agent):
        if self._on_action:
            if self._count % self._periodicity == 0 and self._count > 0:
                self._update(agent)
            self._count += 1

    def onActionChosen(self, agent, action):
        if self._before_action:
            if self._count % self._periodicity == 0 and self._count > 0:
                self._update(agent)
            self._count += 1

    def onEpochEnd(self, agent):
        if self._on_epoch:
            if self._count % self._periodicity == 0 and self._count > 0:
                self._update(agent)
            self._count += 1

    def _update(self, agent):
        reward = agent._dataset.rewards()[-1]
        self._plotter.plot('rewards', np.array([self._count]), np.array([reward]), title_name='Rewards')

        if self._include_secondary_rewards:
            sec_reward = agent._dataset.secondaryRewards()[-1]
            self._plotter.plot('secondary rewards', np.array([self._count]), np.array([sec_reward]),
                               title_name='Secondary Rewards')

class CheckpointController(Controller):
    def __init__(self, evaluate_on='train_loop', periodicity=1,
                 experiment_dir='default', save_dataset=True,
                 save_env=True, **kwargs):
        super(CheckpointController, self).__init__(**kwargs)
        self._on_train_loop = 'train_loop' == evaluate_on
        self._on_train_step = 'train_step' == evaluate_on
        self._before_action = 'before_action' == evaluate_on
        self._on_action = 'action' == evaluate_on
        self._on_episode = 'episode' == evaluate_on
        self._on_epoch = 'epoch' == evaluate_on
        self._evaluate_on = evaluate_on
        self._save_env = save_env

        self._periodicity = periodicity
        self._experiment_dir = experiment_dir
        self._save_dataset = save_dataset

        self._keep_every = kwargs.get('keep_every', 1)
        self._skip_first = kwargs.get('skip_first', 0)
        self._prev_filename = None
        self._updates = 0

    def onTrainLoopTaken(self, agent):
        if self._on_train_loop:
            if self._count > self._skip_first and self._count % self._periodicity == 0 and self._count > 0:
                self._update(agent)
            self._count += 1

    def onActionTaken(self, agent):
        if self._on_action:
            if self._count > self._skip_first and self._count % self._periodicity == 0 and self._count > 0:
                self._update(agent)
            self._count += 1

    def onActionChosen(self, agent, action):
        if self._before_action:
            if self._count > self._skip_first and self._count % self._periodicity == 0 and self._count > 0:
                self._update(agent)
            self._count += 1

    def onEpisodeEnd(self, agent, terminal_reached, reward):
        if self._on_episode:
            if self._count % self._periodicity == 0 and self._count > 0:
                self._update(agent)
            self._count += 1

    def onEnd(self, agent):
        self._update(agent)

    def onEpochEnd(self, agent):
        if self._on_epoch:
            if self._count % self._periodicity == 0 and self._count > 0:
                self._update(agent)
            self._count += 1

    def _update(self, agent):
        print(f"Saving model and agent from {self._evaluate_on} {self._count}")
        model_fname = os.path.join(self._experiment_dir, "model")
        agent.dumpNetwork(model_fname, self._count)
        if ((self._updates - 1) % self._keep_every != 0) and self._prev_filename is not None:
            os.remove(self._prev_filename)

        if self._save_dataset:
            dataset_fname = os.path.join(self._experiment_dir, "dataset")
            agent._dataset.save(dataset_fname, save_env=self._save_env)
        self._prev_filename = f"{model_fname}.epoch={self._count}"
        self._updates += 1


class FindBestController(Controller):
    """A controller that finds the neural net performing at best in validation mode (i.e. for mode = [validationID]) 
    and computes the associated generalization score in test mode (i.e. for mode = [testID], and this only if [testID] 
    is different from None). This controller should never be disabled by InterleavedTestControllers as it is meant to 
    work in conjunction with them.
    
    At each epoch end where this controller is active, it will look at the current mode the agent is in. 
    
    If the mode matches [validationID], it will take the total reward of the agent on this epoch and compare it to its 
    current best score. If it is better, it will ask the agent to dump its current nnet on disk and update its current 
    best score. In all cases, it saves the validation score obtained in a vector.

    If the mode matches [testID], it saves the test (= generalization) score in another vector. Note that if [testID] 
    is None, no test mode score are ever recorded.

    At the end of the experiment (onEnd), if active, this controller will print information about the epoch at which 
    the best neural net was found together with its generalization score, this last information shown only if [testID] 
    is different from None. Finally it will dump a dictionnary containing the data of the plots ({n: number of 
    epochs elapsed, ts: test scores, vs: validation scores}). Note that if [testID] is None, the value dumped for the
    'ts' key is [].
    
    Parameters
    ----------
    validationID : int 
        See synopsis
    testID : int 
        See synopsis
    unique_fname : str
        A unique filename (basename for score and network dumps).
    """

    def __init__(self, validationID=0, testID=None, unique_fname="nnet"):
        super(self.__class__, self).__init__()

        self._validationScores = []
        self._testScores = []
        self._epochNumbers = []
        self._trainingEpochCount = 0
        self._testID = testID
        self._validationID = validationID
        self._filename = unique_fname
        self._bestValidationScoreSoFar = -9999999

    def onEpochEnd(self, agent):
        if (self._active == False):
            return

        mode = agent.mode()
        if mode == self._validationID:
            score, _ = agent.totalRewardOverLastTest()
            self._validationScores.append(score)
            self._epochNumbers.append(self._trainingEpochCount)
            if score > self._bestValidationScoreSoFar:
                self._bestValidationScoreSoFar = score
                print("Dumping Networks")
                agent.dumpNetwork(self._filename, self._trainingEpochCount)
        elif mode == self._testID:
            score, _ = agent.totalRewardOverLastTest()
            self._testScores.append(score)
        else:
            self._trainingEpochCount += 1
        
    def onEnd(self, agent):
        if (self._active == False):
            return

        bestIndex = np.argmax(self._validationScores)
        print("Best neural net obtained after {} epochs, with validation score {}".format(bestIndex+1, self._validationScores[bestIndex]))
        if self._testID != None:
            print("Test score of this neural net: {}".format(self._testScores[bestIndex]))
                
        try:
            os.mkdir("scores")
        except Exception:
            pass
        basename = "scores/" + self._filename
        joblib.dump({"vs": self._validationScores, "ts": self._testScores}, basename + "_scores.jldump")



if __name__ == "__main__":
    pass

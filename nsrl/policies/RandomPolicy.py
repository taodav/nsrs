from ..base_classes import Policy


class RandomPolicy(Policy):
    """The policy acts randomly.

    Parameters
    -----------
    epsilon : float
        Proportion of random steps
    """
    def __init__(self, learning_algo, n_actions, random_state):
        Policy.__init__(self, learning_algo, n_actions, random_state)
        self._epsilon = 1

    def action(self, state, mode=None, *args, **kwargs):
        action, V = self.randomAction()

        return action, V

    def setEpsilon(self, e):
        self._epsilon = e

    def epsilon(self):
        """ Get the epsilon for :math:`\epsilon`-greedy exploration
        """
        return self._epsilon

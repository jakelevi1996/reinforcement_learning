import numpy as np
from agents.bandits.bandit_agent import _BanditAgent

class EpsilonGreedy(_BanditAgent):
    """
    This class represents the epsilon greedy algorithm for the K-armed bandit,
    which estimates each action value using sequential estimation, and upon
    choosing an action makes a random choice with probability epsilon (to
    encourage exploration), and makes a greedy action choice (choosing the
    action with the highest value) with probability 1 - epsilon
    """
    def __init__(
        self,
        epsilon=0.02,
        step_size=0.3,
        num_actions=10,
        initial_value_estimates=None,
        rng=None,
    ):
        self._epsilon = epsilon
        self._step_size = step_size
        self._num_action_tries = np.zeros(num_actions, dtype=np.int)

        if initial_value_estimates is None:
            self._value_estimates = np.zeros(num_actions)
        else:
            self._value_estimates = initial_value_estimates

        if rng is None:
            self._rng = np.random.default_rng()
        else:
            self._rng = rng

    def choose_action(self):
        is_greedy = (self._rng.random() > self._epsilon)
        if is_greedy:
            optimal_value = max(self._value_estimates)
            optimal_action_list = [
                a
                for a, value in enumerate(self._value_estimates)
                if value == optimal_value
            ]
            if len(optimal_action_list) == 1:
                action = optimal_action_list[0]
            else:
                action = self._rng.choice(optimal_action_list)
        else:
            action = self._rng.integers(self._value_estimates.size)

        self._num_action_tries[action] += 1
        return action

    def update(self, action, reward):
        self._value_estimates[action] += (
            (reward - self._value_estimates[action])
            / self._num_action_tries[action]
        )

    def get_name(self):
        name = "$\\varepsilon$-greedy$(\\varepsilon=%.2f)$" % self._epsilon
        return name

class EpsilonGreedyConstantStepSize(EpsilonGreedy):
    """
    This class represents the epsilon greedy algorithm for the K-armed bandit,
    which estimates each action value using a constant step size, and upon
    choosing an action makes a random choice with probability epsilon (to
    encourage exploration), and makes a greedy action choice (choosing the
    action with the highest value) with probability 1 - epsilon
    """
    def update(self, action, reward):
        self._value_estimates[action] += (
            self._step_size * (reward - self._value_estimates[action])
        )

    def get_name(self):
        name = (
            "$\\varepsilon$-greedy$(\\varepsilon=%.2f,\\alpha=%.2f)$"
            % (self._epsilon, self._step_size)
        )
        return name

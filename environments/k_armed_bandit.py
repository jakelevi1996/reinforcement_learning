import numpy as np

class KArmedBandit:
    def __init__(self, k=10, rng=None, mean_reward=0):
        """
        Initialise a K-armed Bandit environment, with the following arguments:

        - k: (optional) number of actions that can be taken
        - rng: (optional) random number generator (useful for repeatable
          experiments)
        - mean_reward: (optional) mean of the distribution from which
          action values are generated (useful because some types of bandit
          algorithms perform worse for different values of mean_reward)
        """
        if rng is None:
            self._rng = np.random.default_rng()
        else:
            self._rng = rng

        self._action_values = self._rng.normal(loc=mean_reward, size=k)
        self._optimal_actions = [
            a for a, value in enumerate(self._action_values)
            if value == max(self._action_values)
        ]

    def step(self, action):
        """
        Given an action, return a reward
        """
        action_value = self._action_values[action]
        reward = self._rng.normal(action_value)
        return reward

    def is_optimal_action(self, action):
        """
        Given an action, return True if this action is optimal, otherwise
        return False (useful for measuring how frequently a given bandit
        algorithm selects the optimal action)
        """
        return (action in self._optimal_actions)

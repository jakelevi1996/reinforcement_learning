import numpy as np

class KArmedBandit:
    def __init__(self, k=10, rng=None, mean_reward=0):
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
        action_value = self._action_values[action]
        reward = self._rng.normal(action_value)
        return reward

    def is_optimal_action(self, action):
        return (action in self._optimal_actions)

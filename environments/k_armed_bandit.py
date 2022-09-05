import numpy as np

class KArmedBandit:
    def __init__(self, k=10, rng=None):
        if rng is None:
            self._rng = np.random.default_rng()
        else:
            self._rng = rng
        self._action_values = self._rng.normal(0, 1, k)

    def step(self, action):
        action_value = self._action_values[action]
        reward = self._rng.normal(action_value, 1, 1)
        return reward

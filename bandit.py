import numpy as np

class KArmedBandit:
    def __init__(self, k=10):
        self._action_values = np.random.normal(0, 1, k)

    def step(self, action):
        action_value = self._action_values[action]
        reward = np.random.normal(action_value, 1, 1)
        return reward

b = KArmedBandit()
print(b._action_values)
print(b.step(3))


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

class EpsilonGreedy:
    def __init__(
        self,
        epsilon=0.1,
        step_size=0.1,
        num_actions=10,
        initial_value_estimates=None,
        rng=None,
    ):
        self._epsilon = epsilon
        self._step_size = step_size
        self._num_action_tries = np.zeros(num_actions)

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
            self._action = self._rng.choice(optimal_action_list)
        else:
            self._action = np.random.randint(self._value_estimates.size)

        return self._action


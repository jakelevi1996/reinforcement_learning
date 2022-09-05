import numpy as np

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
        self._num_action_tries = np.zeros(num_actions, dtype=np.int)

        if initial_value_estimates is None:
            self._value_estimates = np.zeros(num_actions)
        else:
            self._value_estimates = initial_value_estimates

        if rng is None:
            self._rng = np.random.default_rng()
        else:
            self._rng = rng

    def get_name(self):
        name = "$\\varepsilon$-greedy$(\\varepsilon=%.2f)$" % self._epsilon
        return name

    def choose_action(self):
        is_greedy = (self._rng.random() > self._epsilon)
        if is_greedy:
            optimal_value = max(self._value_estimates)
            optimal_action_list = [
                a
                for a, value in enumerate(self._value_estimates)
                if value == optimal_value
            ]
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

class EpsilonGreedyConstantStepSize(EpsilonGreedy):
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

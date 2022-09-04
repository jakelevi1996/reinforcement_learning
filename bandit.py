import numpy as np

class KArmedBandit:
    def __init__(self, k=10):
        self._action_values = np.random.normal(0, 1, k)

    def step(self, action):
        action_value = self._action_values[action]
        reward = np.random.normal(action_value, 1, 1)
        return reward

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
            action = np.random.randint(self._value_estimates.size)

        self._num_action_tries[action] += 1
        return action

    def update(self, action, reward):
        self._value_estimates[action] += (
            (reward - self._value_estimates[action])
            / self._num_action_tries[action]
        )


env = KArmedBandit()
print(env._action_values)
print(env.step(3))
agent = EpsilonGreedy()
N = 1000
for i in range(N):
    print("Performing step %i/%i" % (i + 1, N), end="\r")
    action = agent.choose_action()
    reward = env.step(action)
    agent.update(action, reward)

optimal_value = max(env._action_values)
optimal_actions = [
    a
    for a, value in enumerate(env._action_values)
    if value == optimal_value
]
percent_optimal = (
    (100.0 * sum(agent._num_action_tries[i] for i in optimal_actions)) / N
)
print(
    "\n",
    env._action_values,
    agent._value_estimates,
    agent._num_action_tries,
    "Percent optimal = %.1f%%" % percent_optimal,
    sep="\n",
)

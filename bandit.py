import numpy as np
import plotting

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
            action = self._rng.integers(self._value_estimates.size)

        self._num_action_tries[action] += 1
        return action

    def update(self, action, reward):
        self._value_estimates[action] += (
            (reward - self._value_estimates[action])
            / self._num_action_tries[action]
        )


num_steps = 1000
num_repeats = 100
reward_array = np.zeros([num_repeats, num_steps])
optimal_choice_array = np.zeros([num_repeats, num_steps])
for i in range(num_repeats):
    if ((i + 1) % 10) == 0:
        print("Performing repeat %i/%i" % (i + 1, num_repeats), end="\r")
    env = KArmedBandit()
    agent = EpsilonGreedy()
    optimal_actions = [
        a for a, value in enumerate(env._action_values)
        if value == max(env._action_values)
    ]

    for j in range(num_steps):
        action = agent.choose_action()
        reward = env.step(action)
        agent.update(action, reward)
        reward_array[i, j] = reward
        if action in optimal_actions:
            optimal_choice_array[i, j] = 1

print("\nPlotting results...")
t = np.arange(num_steps)
t_tiled = np.tile(t.reshape(1, -1), [num_repeats, 1])
alpha = 0.5/num_repeats
plotting.plot(
    [
        plotting.Line(t_tiled, reward_array, "b", "", "o", alpha, 10),
        plotting.Line(t, np.mean(reward_array, axis=0), "b", "-", "", 1, 20),
    ],
    "Epsilon greedy rewards",
    axis_properties=plotting.AxisProperties("Time", "Reward", None, [-2, 4]),
)
percent_optimal_choice = 100*np.mean(optimal_choice_array, axis=0)
plotting.plot(
    [plotting.Line(t, percent_optimal_choice, "b", "-", "", 1, 20)],
    "Epsilon greedy percentage of optimal actions",
    axis_properties=plotting.AxisProperties(
        "Time",
        "% Optimal action",
        None,
        [0, 100],
    ),
)

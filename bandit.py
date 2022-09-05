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

class AgentResult:
    def __init__(self, agent_type, name, num_steps, num_repeats):
        self.construcor = agent_type
        self.name = name
        self.reward_array = np.zeros([num_repeats, num_steps])
        self.optimal_choice_array = np.zeros([num_repeats, num_steps])

    def get_mean_std_reward(self):
        self.mean_reward = np.mean(self.reward_array, axis=0)
        self.std_reward = np.std(self.reward_array, axis=0)

num_steps = 1000
num_repeats = 100
agent_result_list = [
    AgentResult(agent_type, agent_type().get_name(), num_steps, num_repeats)
    for agent_type in [EpsilonGreedy, EpsilonGreedyConstantStepSize]
]
for i in range(num_repeats):
    if ((i + 1) % 10) == 0:
        print("Performing repeat %i/%i" % (i + 1, num_repeats), end="\r")
    env = KArmedBandit()
    optimal_actions = [
        a for a, value in enumerate(env._action_values)
        if value == max(env._action_values)
    ]
    for agent_result in agent_result_list:
        agent = agent_result.construcor()

        for j in range(num_steps):
            action = agent.choose_action()
            reward = env.step(action)
            agent.update(action, reward)
            agent_result.reward_array[i, j] = reward
            if action in optimal_actions:
                agent_result.optimal_choice_array[i, j] = 1

print("\nPlotting results...")
t = np.arange(num_steps)
mt = 0.5 / num_repeats
line_props = {"ls": "-", "marker": "", "alpha": 1, "zorder": 20}
marker_props = {"ls": "", "marker": "o", "alpha": mt, "zorder": 10}
cp = plotting.ColourPicker(len(agent_result_list))
for a in agent_result_list:
    a.get_mean_std_reward()
rewards_line_list = [
    plotting.Line(
        t,
        agent_result.reward_array.T,
        color=cp(i),
        label="%s (single reward)" % agent_result.name,
        **marker_props,
    )
    for i, agent_result in enumerate(agent_result_list)
]
mean_reward_line_list = [
    plotting.Line(
        t,
        agent_result.mean_reward,
        color=cp(i),
        label="%s (mean reward)" % agent_result.name,
        **line_props,
    )
    for i, agent_result in enumerate(agent_result_list)
]
std_reward_fb_list = [
    plotting.FillBetween(
        t,
        agent_result.mean_reward + agent_result.std_reward,
        agent_result.mean_reward - agent_result.std_reward,
        color=cp(i),
        label="$\\pm\\sigma$",
        alpha=0.2,
    )
    for i, agent_result in enumerate(agent_result_list)
]
percent_optimal_choice_line_list = [
    plotting.Line(
        t,
        100 * np.mean(agent_result.optimal_choice_array, axis=0),
        color=cp(i),
        label=agent_result.name,
        **line_props,
    )
    for i, agent_result in enumerate(agent_result_list)
]
plotting.plot(
    [
        line
        for line_pair in zip(rewards_line_list, mean_reward_line_list)
        for line in line_pair
    ],
    "Epsilon greedy rewards",
    axis_properties=plotting.AxisProperties("Time", "Reward", None, [-2, 4]),
    legend_properties=plotting.LegendProperties(0.4),
    figsize=[12, 6],
)
plotting.plot(
    [
        line
        for line_pair in zip(mean_reward_line_list, std_reward_fb_list)
        for line in line_pair
    ],
    "Epsilon greedy rewards (mean and variance)",
    axis_properties=plotting.AxisProperties("Time", "Reward", None, [-1, 4]),
    legend_properties=plotting.LegendProperties(0.4),
    figsize=[12, 6],
)
plotting.plot(
    percent_optimal_choice_line_list,
    "Epsilon greedy percentage of optimal actions",
    axis_properties=plotting.AxisProperties(
        "Time",
        "% Optimal action",
        None,
        [0, 100],
    ),
    legend_properties=plotting.LegendProperties(),
)

import argparse
import os
import time
import numpy as np
if __name__ == "__main__":
    import __init__
import agents
import environments
import plotting
import util

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

class AgentResult:
    def __init__(self, agent_type, name, num_steps, num_repeats):
        self.construcor = agent_type
        self.name = name
        self.reward_array = np.zeros([num_repeats, num_steps])
        self.optimal_choice_array = np.zeros([num_repeats, num_steps])

    def get_mean_std_reward(self):
        self.mean_reward = np.mean(self.reward_array, axis=0)
        self.std_reward = np.std(self.reward_array, axis=0)

def main(agent_result_list, args):
    for i in range(args.num_repeats):
        if ((i + 1) % 10) == 0:
            print(
                "Performing repeat %i/%i..."
                % (i + 1, args.num_repeats), end="\r"
            )
        env = environments.KArmedBandit()
        optimal_actions = [
            a for a, value in enumerate(env._action_values)
            if value == max(env._action_values)
        ]
        for agent_result in agent_result_list:
            agent = agent_result.construcor()

            for j in range(args.num_steps):
                action = agent.choose_action()
                reward = env.step(action)
                agent.update(action, reward)
                agent_result.reward_array[i, j] = reward
                if action in optimal_actions:
                    agent_result.optimal_choice_array[i, j] = 1

def plot(agent_result_list, args):
    t = np.arange(args.num_steps)
    mt = 0.5 / args.num_repeats
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
        (
            "Epsilon greedy rewards (%i steps, %i repeats)"
            % (args.num_steps, args.num_repeats)
        ),
        args.results_dir,
        axis_properties=plotting.AxisProperties(
            "Time",
            "Reward",
            None,
            [-2, 4],
        ),
        legend_properties=plotting.LegendProperties(0.4),
        figsize=[12, 6],
    )
    plotting.plot(
        [
            line
            for line_pair in zip(mean_reward_line_list, std_reward_fb_list)
            for line in line_pair
        ],
        (
            "Epsilon greedy rewards (mean and variance, %i steps, %i repeats)"
            % (args.num_steps, args.num_repeats)
        ),
        args.results_dir,
        axis_properties=plotting.AxisProperties(
            "Time",
            "Reward",
            None,
            [-1, 4],
        ),
        legend_properties=plotting.LegendProperties(0.4),
        figsize=[12, 6],
    )
    plotting.plot(
        percent_optimal_choice_line_list,
        (
            "Epsilon greedy percentage of optimal actions "
            "(%i steps, %i repeats)"
            % (args.num_steps, args.num_repeats)
        ),
        args.results_dir,
        axis_properties=plotting.AxisProperties(
            "Time",
            "% Optimal action",
            None,
            [0, 100],
        ),
        legend_properties=plotting.LegendProperties(),
    )

if __name__ == "__main__":
    # Define CLI using argparse
    parser = argparse.ArgumentParser(description="Compare bandit algorithms")

    parser.add_argument(
        "--results_dir",
        help="Name of directory in which results should be saved",
        default=None,
        type=str,
    )
    parser.add_argument(
        "--save_data_filename",
        help="Filename in which new results should be saved (this argument is "
        "ignored if --load_data_filename is present)",
        default=None,
        type=str,
    )
    parser.add_argument(
        "--load_data_filename",
        help="If present, do not perform any new experiments, and instead "
        "load data from the specified filename and plot the results",
        default=None,
        type=str,
    )
    parser.add_argument(
        "--num_steps",
        help="Number of time steps to simulate for each rollout",
        default=1000,
        type=int,
    )
    parser.add_argument(
        "--num_repeats",
        help="Number of different environments in which to test each agent",
        default=100,
        type=int,
    )

    # Parse arguments
    args = parser.parse_args()

    # If we're loading data from file, do so now, because in case
    # args.results_dir hasn't been provided, args.num_steps and
    # args.num_repeats need to be loaded before args.results_dir is set
    if args.load_data_filename is not None:
        result_data = util.Result(args.load_data_filename).load()
        agent_result_list, args.num_steps, args.num_repeats = result_data

    if args.results_dir is None:
        args.results_dir = os.path.join(
            CURRENT_DIR,
            "Results",
            "Bandit",
            "%i_repeats_%i_steps" % (args.num_repeats, args.num_steps)
        )

    if args.load_data_filename is None:
        if args.save_data_filename is None:
            args.save_data_filename = os.path.join(
                args.results_dir,
                "bandit_data.pkl",
            )
        agent_result_list = [
            AgentResult(
                agent_type,
                agent_type().get_name(),
                args.num_steps,
                args.num_repeats,
            )
            for agent_type in [
                agents.bandits.EpsilonGreedy,
                agents.bandits.EpsilonGreedyConstantStepSize,
            ]
        ]
        result_data = [agent_result_list, args.num_steps, args.num_repeats]
        result = util.Result(args.save_data_filename, result_data)
        with result.get_results_saving_context():
            t_start = time.perf_counter()

            main(agent_result_list, args)

            t_total = time.perf_counter() - t_start
            print("\nFinished main function in %.1fs" % t_total)

    print("Plotting results...")
    plot(agent_result_list, args)

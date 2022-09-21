"""
Sweep over a range of parameters for each bandit algorithm to find the
approximately optimal parameters, and plot the results of each parameter sweep
on a graph which is saved to disk
"""
import argparse
import os
import time
import numpy as np
if __name__ == "__main__":
    import __init__
import agents
import environments
import sweep
import util

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

class _TestBanditAgent(sweep.Experiment):
    def get_agent(self, **kwargs):
        raise NotImplementedError()

    def __init__(self, num_steps, seeder):
        self._num_steps = num_steps
        self._seeder = seeder

    def run(self, **kwargs):
        env = environments.KArmedBandit()
        agent = self.get_agent(**kwargs)
        total_reward = 0

        for _ in range(self._num_steps):
            action = agent.choose_action()
            reward = env.step(action)
            agent.update(action, reward)
            total_reward += reward

        mean_reward = total_reward / self._num_steps
        return mean_reward

class TestEpsilonGreedy(_TestBanditAgent):
    def get_agent(self, epsilon):
        return agents.bandits.EpsilonGreedy(epsilon)

class TestEpsilonGreedyConstantStepSize(_TestBanditAgent):
    def get_agent(self, epsilon, step_size):
        agent = agents.bandits.EpsilonGreedyConstantStepSize(
            epsilon,
            step_size,
        )
        return agent

class TestGradientBandit(_TestBanditAgent):
    def get_agent(self, step_size):
        return agents.bandits.GradientBandit(step_size)

def main(args):
    filename_list_list = [
        test_epsilon_greedy(args),
        test_epsilon_greedy_constant_step_size(args),
        test_gradient_bandit(args),
    ]
    print("\nPlots saved with the following filenames:\n")
    for filename_list in filename_list_list:
        print(*filename_list, sep="\n", end="\n\n")


def test_epsilon_greedy(args):
    seeder = util.Seeder()
    experiment = TestEpsilonGreedy(args.num_steps, seeder)
    param_sweeper = sweep.ParamSweeper(
        experiment,
        n_repeats=args.num_repeats,
        print_every=50,
    )

    param_sweeper.add_parameter(
        sweep.Parameter(
            "epsilon",
            0.1,
            val_lo=0.01,
            val_hi=0.6,
            val_num=args.num_values,
            log_space=True,
        )
    )
    param_sweeper.find_best_parameters()
    results_dir = os.path.join(args.results_dir, "Epsilon_greedy")
    return param_sweeper.plot("Epsilon greedy", results_dir)

def test_epsilon_greedy_constant_step_size(args):
    seeder = util.Seeder()
    experiment = TestEpsilonGreedyConstantStepSize(args.num_steps, seeder)
    param_sweeper = sweep.ParamSweeper(
        experiment,
        n_repeats=args.num_repeats,
        print_every=50,
    )

    param_sweeper.add_parameter(
        sweep.Parameter(
            "epsilon",
            0.1,
            val_lo=0.01,
            val_hi=0.6,
            val_num=args.num_values,
            log_space=True,
        )
    )
    param_sweeper.add_parameter(
        sweep.Parameter(
            "step_size",
            0.1,
            val_lo=0.01,
            val_hi=1,
            val_num=args.num_values,
            log_space=True,
        )
    )
    param_sweeper.find_best_parameters()
    results_dir = os.path.join(
        args.results_dir,
        "Epsilon_greedy_constant_step_size",
    )
    experiment_name = "Epsilon greedy (constant step size)"
    return param_sweeper.plot(experiment_name, results_dir)

def test_gradient_bandit(args):
    seeder = util.Seeder()
    experiment = TestGradientBandit(args.num_steps, seeder)
    param_sweeper = sweep.ParamSweeper(
        experiment,
        n_repeats=args.num_repeats,
        print_every=50,
    )

    param_sweeper.add_parameter(
        sweep.Parameter(
            "step_size",
            0.1,
            val_lo=0.01,
            val_hi=1,
            val_num=args.num_values,
            log_space=True,
        )
    )
    param_sweeper.find_best_parameters()
    results_dir = os.path.join(
        args.results_dir,
        "Gradient_bandit",
    )
    experiment_name = "Gradient bandit"
    return param_sweeper.plot(experiment_name, results_dir)

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
    parser.add_argument(
        "--num_values",
        help="Number of different values to test for each parameter",
        default=20,
        type=int,
    )

    # Parse arguments
    args = parser.parse_args()

    if args.results_dir is None:
        args.results_dir = os.path.join(
            CURRENT_DIR,
            "Results",
            "Param_sweeps",
            "Bandit",
            "%i_repeats_%i_steps_%i_values"
            % (args.num_repeats, args.num_steps, args.num_values),
        )

    util.time_func(main, args)

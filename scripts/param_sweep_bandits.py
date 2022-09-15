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

class TestEpsilonGreedy(sweep.Experiment):
    def __init__(self, num_steps, seeder):
        self._num_steps = num_steps
        self._seeder = seeder

    def run(self, epsilon, step_size):
        env = environments.KArmedBandit()
        agent = agents.bandits.EpsilonGreedy(epsilon, step_size)
        total_reward = 0

        for _ in range(self._num_steps):
            action = agent.choose_action()
            reward = env.step(action)
            agent.update(action, reward)
            total_reward += reward

        mean_reward = total_reward / self._num_steps
        return mean_reward

def main(args):
    test_epsilon_greedy(args)

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
            log_space=True,
        )
    )
    param_sweeper.add_parameter(
        sweep.Parameter(
            "step_size",
            0.1,
            val_lo=0.01,
            val_hi=1,
            log_space=True,
        )
    )
    param_sweeper.find_best_parameters()
    results_dir = os.path.join(args.results_dir, "Epsilon_greedy")
    param_sweeper.plot(results_dir)

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

    # Parse arguments
    args = parser.parse_args()

    if args.results_dir is None:
        args.results_dir = os.path.join(
            CURRENT_DIR,
            "Results",
            "Param_sweeps",
            "Bandit",
            "%i_repeats_%i_steps" % (args.num_repeats, args.num_steps)
        )

    util.time_func(main, args)

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
RESULTS_DIR = os.path.join(CURRENT_DIR, "Results", "Param_sweeps", "Bandits")

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

seeder = util.Seeder()
experiment = TestEpsilonGreedy(1000, seeder)
param_sweeper = sweep.ParamSweeper(experiment, n_repeats=100, print_every=50)

param_sweeper.add_parameter(
    sweep.Parameter("epsilon", 0.1, val_lo=0.01, val_hi=0.6, log_space=True)
)
param_sweeper.add_parameter(
    sweep.Parameter("step_size", 0.1, val_lo=0.01, val_hi=1, log_space=True)
)
param_sweeper.find_best_parameters()
param_sweeper.plot(RESULTS_DIR)

import numpy as np
from agents.bandits.bandit_agent import _BanditAgent

class GradientBandit(_BanditAgent):
    """
    This class represents a bandit algorithm that estimates a numerical
    preference for each action, which is optimised using stochastic gradient
    ascent on the expected rewards. Actions are sampled with probabilies
    calculated by applying a softmax function to the action preferences.
    """
    def __init__(self, step_size=0.5, num_actions=10, rng=None):
        self._num_actions = num_actions
        self._step_size = step_size
        self._step = 1
        self._mean_reward = 0
        self._action_preferences = np.zeros(num_actions)

        if rng is None:
            self._rng = np.random.default_rng()
        else:
            self._rng = rng

    def choose_action(self):
        e = np.exp(self._action_preferences)
        self._p = e / sum(e)
        action = self._rng.choice(self._num_actions, p=self._p)
        return action

    def update(self, action, reward):
        self._mean_reward += (reward - self._mean_reward) / self._step
        self._step += 1
        inc = self._step_size * (reward - self._mean_reward)
        self._action_preferences[action] += inc
        self._action_preferences -= inc * self._p

    def get_name(self):
        name = "Gradient bandit$(\\alpha=%.2f)$" % self._step_size
        return name

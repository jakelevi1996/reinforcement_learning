import numpy as np
from agents.bandits.bandit_agent import _BanditAgent

class _BayesianSampler(_BanditAgent):
    def __init__(self, num_actions=10, rng=None):
        self._step = 0
        self._prior_mean = 0
        self._prior_mean_square = 0
        self._prior_var = 0
        self._num_actions = num_actions
        self._num_action_tries = np.zeros(num_actions, dtype=np.int)
        self._mean_reward = np.zeros(num_actions)
        self._mean_square_reward = np.zeros(num_actions)
        self._var_reward = np.ones(num_actions)

        if rng is None:
            self._rng = np.random.default_rng()
        else:
            self._rng = rng

    def choose_action(self):

        if self._prior_var == 0:
            return self._rng.choice(self._num_actions)

        value_posterior_var = (
            1.0 / (
                (self._num_action_tries / self._var_reward)
                + (1.0 / self._prior_var)
            )
        )
        value_posterior_mean = (
            (
                (self._num_action_tries * self._mean_reward / self._var_reward)
                + (self._prior_mean / self._prior_var)
            )
            * value_posterior_var
        )

        samples = self._rng.normal(
            value_posterior_mean,
            np.sqrt(value_posterior_var),
        )
        action = np.argmax(samples)
        return action

    def update(self, action, reward):
        self._num_action_tries[action] += 1
        self._mean_reward[action] += (
            (reward - self._mean_reward[action])
            / self._num_action_tries[action]
        )
        self._mean_square_reward[action] += (
            ((reward * reward) - self._mean_square_reward[action])
            / self._num_action_tries[action]
        )
        self._var_reward[action] = (
            self._mean_square_reward[action]
            - np.square(self._mean_reward[action])
        )
        if self._var_reward[action] == 0:
            self._var_reward[action] = np.var(self._mean_reward)

        self._set_prior(reward)

    def _set_prior(self, reward):
        raise NotImplementedError

class BayesianSamplerBroadPrior(_BayesianSampler):
    def _set_prior(self, reward):
        self._step += 1
        self._prior_mean += (reward - self._prior_mean) / self._step
        self._prior_mean_square += (
            ((reward * reward) - self._prior_mean_square)
            / self._step
        )
        self._prior_var = (
            self._prior_mean_square
            - np.square(self._prior_mean)
        )

    def get_name(self):
        return "Bayesian sampler (broad prior)"

class BayesianSamplerValuePrior(_BayesianSampler):
    def _set_prior(self, reward):
        self._prior_mean = np.mean(self._mean_reward)
        self._prior_var = np.var(self._mean_reward)

    def get_name(self):
        return "Bayesian sampler (value prior)"

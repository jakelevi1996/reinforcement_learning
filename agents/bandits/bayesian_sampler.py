import numpy as np
from agents.bandits.bandit_agent import _BanditAgent

class _BayesianSampler(_BanditAgent):
    def __init__(self, num_actions=10, rng=None):
        self._step = 0
        self._num_actions = num_actions
        self._num_action_tries = np.zeros(num_actions, dtype=np.int)
        self._prior_mean = 0
        self._prior_mean_square = 0
        self._prior_var = 0
        self._likelihood_mean = np.zeros(num_actions)
        self._likelihood_mean_square = np.zeros(num_actions)
        self._likelihood_var = np.ones(num_actions)

        if rng is None:
            self._rng = np.random.default_rng()
        else:
            self._rng = rng

    def choose_action(self):

        if self._prior_var == 0:
            return self._rng.choice(self._num_actions)

        posterior_var = (
            1.0 / (
                (self._num_action_tries / self._likelihood_var)
                + (1.0 / self._prior_var)
            )
        )
        posterior_mean = (
            (
                (
                    self._num_action_tries * self._likelihood_mean
                    / self._likelihood_var
                )
                + (self._prior_mean / self._prior_var)
            )
            * posterior_var
        )

        samples = self._rng.normal(posterior_mean, np.sqrt(posterior_var))
        action = np.argmax(samples)
        return action

    def update(self, action, reward):
        self._num_action_tries[action] += 1
        self._likelihood_mean[action] += (
            (reward - self._likelihood_mean[action])
            / self._num_action_tries[action]
        )
        self._likelihood_mean_square[action] += (
            ((reward * reward) - self._likelihood_mean_square[action])
            / self._num_action_tries[action]
        )
        self._likelihood_var[action] = (
            self._likelihood_mean_square[action]
            - np.square(self._likelihood_mean[action])
        )
        if self._likelihood_var[action] == 0:
            self._likelihood_var[action] = np.var(self._likelihood_mean)

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
        self._prior_mean = np.mean(self._likelihood_mean)
        self._prior_var = np.var(self._likelihood_mean)

    def get_name(self):
        return "Bayesian sampler (value prior)"

import numpy as np
from agents.bandits.bandit_agent import _BanditAgent

class _BayesianSampler(_BanditAgent):
    """
    This abstract class represents a bandit algorithm in which:

    - A Gaussian posterior distribution is maintained over each action value,
      which is proportional to the product of the likelihood and prior
      distributions
        - Note that the distribution over the value (mean reward) of an action
          accounts for epistemic uncertainty but not alleatoric uncertainty,
          which is different to the distribution over the reward returned from
          an action, which accounts for both epistemic and alleatoric
          uncertainty
    - A Gaussian likelihood distribution is maintained over each action value,
      and is updated each time a reward is received using sequential estimation
    - A Gaussian prior distribution is shared among all action values, and is
      updated each time a reward is received
    - Actions are chosen by drawing a sample from the estimated posterior
      distribution over each action value, and choosing the action whose value
      corresponds to the highest sample
    """
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
        """
        Set the shared prior distribution over all action values as the maximum
        likelihood Gaussian distribution over all rewards received from all
        actions. Note that this biases the prior distribution towards values of
        actions which are taken more often, which are generally those actions
        with higher estimated action values, leading to an optimistic prior
        distribution over action values which may encourage exploration
        """
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
        """
        Set the shared prior distribution over all action values as the maximum
        likelihood Gaussian distribution of the means of the estimated
        likelihood distributions of all action values. Note that:

        - If the means of the estimated likelihood distributions of all action
          values span a wide range, then the prior distribution over action
          values will have a large variance
        - Because the means of the estimated likelihood distributions of all
          the action values are estimated using all the rewards that have been
          received, this is similar to setting the prior distribution over
          action values equal to maximum likelihood Gaussian distribution over
          all rewards received from all actions. The main difference is that
          each action contributes equally to the prior distribution in this
          case, even if an action has not been sampled many times (which would
          generally imply that the action has a lower value), which biases the
          prior relatively more towards actions with lower action values, which
          relatively discourages exploration and encourages exploitation
        """
        self._prior_mean = np.mean(self._likelihood_mean)
        self._prior_var = np.var(self._likelihood_mean)

    def get_name(self):
        return "Bayesian sampler (value prior)"

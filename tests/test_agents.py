import pytest
import agents

bandit_agent_list = [
    agents.bandits.EpsilonGreedy,
    agents.bandits.EpsilonGreedyConstantStepSize,
    agents.bandits.GradientBandit,
    agents.bandits.BayesianSamplerValuePrior,
    agents.bandits.BayesianSamplerBroadPrior,
]

@pytest.mark.parametrize("bandit_type", bandit_agent_list)
def test_bandit_agents(bandit_type):
    pass

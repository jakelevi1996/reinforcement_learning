import numpy as np
import pytest
import tests.util
import util
import agents

OUTPUT_DIR = tests.util.get_output_dir("test_agents")

bandit_agent_list = [
    agents.bandits.EpsilonGreedy,
    agents.bandits.EpsilonGreedyConstantStepSize,
    agents.bandits.GradientBandit,
    agents.bandits.BayesianSamplerValuePrior,
    agents.bandits.BayesianSamplerBroadPrior,
]

@pytest.mark.parametrize("bandit_type", bandit_agent_list)
def test_bandit_agents(bandit_type):
    printer = util.Printer("%s.txt" % (bandit_type.__name__), OUTPUT_DIR)
    seed = util.Seeder().get_seed("test_bandit_agents", bandit_type)
    rng = np.random.default_rng(seed)
    printer.print("Seed = %i" % seed)

    num_actions = rng.choice(range(5, 15))
    printer.print("num_actions = %i" % num_actions)

    agent = bandit_type(num_actions=num_actions, rng=rng)
    printer.print("agent name = %s" % agent.get_name())

    for i in range(rng.choice(range(10, 20))):
        action = agent.choose_action()
        assert (action >= 0) and (action < num_actions)
        printer.print("Action %i = %s" % (i, action))
        reward = rng.normal()
        agent.update(action, reward)

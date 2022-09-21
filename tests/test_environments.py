import numpy as np
import pytest
import tests.util
import util
import environments

OUTPUT_DIR = tests.util.get_output_dir("test_environments")

@pytest.mark.parametrize("repeat", range(3))
def test_bandit_environment(repeat):
    """
    Test the environments.KArmedBandit class, including the step and
    is_optimal_action methods
    """
    printer = util.Printer("%s %i.txt" % ("KArmedBandit", repeat), OUTPUT_DIR)
    seed = util.Seeder().get_seed("KArmedBandit", repeat)
    rng = np.random.default_rng(seed)
    printer.print("Seed = %i" % seed)

    num_actions = rng.choice(range(5, 25))
    printer.print("num_actions = %i" % num_actions)

    env = environments.KArmedBandit(k=num_actions, rng=rng)

    # Test each action sequentially
    for action in range(num_actions):
        reward = env.step(action)
        optimal = env.is_optimal_action(action)
        printer.print(
            "action = %i, reward = %.1f, optimal = %s"
            % (action, reward, optimal)
        )

    # Test some random choices of actions
    for i in range(rng.choice(range(10, 20))):
        action = rng.choice(num_actions)
        reward = env.step(action)
        optimal = env.is_optimal_action(action)
        printer.print(
            "action = %i, reward = %.1f, optimal = %s"
            % (action, reward, optimal)
        )

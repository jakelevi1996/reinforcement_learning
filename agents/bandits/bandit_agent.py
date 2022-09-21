class _BanditAgent:
    """
    Abstract parent class for bandit agents
    """
    def choose_action(self):
        """
        Choose an action to take
        """
        raise NotImplementedError

    def update(self, action, reward):
        """
        Given an action that was taken, and the reward that was returned by the
        environment when that action was taken, update the internal parameters
        of this bandit agent object
        """
        raise NotImplementedError

    def get_name(self):
        """
        Return a string representing this bandit agent, which is used for
        example in legend entries
        """
        raise NotImplementedError

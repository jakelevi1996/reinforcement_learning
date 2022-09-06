class _BanditAgent:
    def get_name(self):
        raise NotImplementedError

    def choose_action(self):
        raise NotImplementedError

    def update(self, action, reward):
        raise NotImplementedError

class _BanditAgent:
    def choose_action(self):
        raise NotImplementedError

    def update(self, action, reward):
        raise NotImplementedError

    def get_name(self):
        raise NotImplementedError

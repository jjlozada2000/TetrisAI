class Agent:
    def __init__(self, bot="tree", weights=None, depth=2, beam=8, discount=0.97):
        self.bot = bot
        self.weights = weights or {}
        self.depth = 1 if bot == "greedy" else depth
        self.beam = beam
        self.discount = discount

    def choose_action(self, env):
        return env.get_best_action(
            bot=self.bot,
            weights=self.weights,
            depth=self.depth,
            beam=self.beam,
            discount=self.discount,
        )
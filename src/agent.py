
from dataclasses import dataclass
from model import load_weights

@dataclass
class SearchConfig:
    depth: int = 2
    take: int = 8
    discount: float = 0.985
    use_hold: bool = True

class SearchAgent:
    def __init__(self, weights=None, config=None):
        self.weights = weights or load_weights()
        self.config = config or SearchConfig()

    def choose_action(self, env):
        return env.get_best_action(
            weights=self.weights,
            depth=self.config.depth,
            take=self.config.take,
            discount=self.config.discount,
            use_hold=self.config.use_hold,
        )

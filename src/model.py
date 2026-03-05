"""
model.py — Deep Q-Network for Tetris

Matching vietnh1009/Tetris-deep-Q-learning-pytorch:
    Input:  4 features (lines_cleared, holes, bumpiness, total_height)
    Hidden: 64 → 64 neurons with ReLU
    Output: 1 scalar (predicted value of this board state)
"""

import torch
import torch.nn as nn


class DeepQNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.net(x)


if __name__ == "__main__":
    model = DeepQNetwork()
    x = torch.FloatTensor([[0, 0, 0, 0]])
    print(f"Output: {model(x).item():.4f}")
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")
    print("OK")
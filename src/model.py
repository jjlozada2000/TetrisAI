"""
model.py — Neural Network for Tetris DQN (nuno-faria style)

Architecture (matching the proven 800K+ point model):
    Input:  4 features (lines_cleared, holes, bumpiness, total_height)
    Hidden: 32 → 32 neurons with ReLU
    Output: 1 scalar — predicted board quality

Only ~1.2K parameters. Tiny network that learns fast.
"""

import torch
import torch.nn as nn


class TetrisNet(nn.Module):
    def __init__(self, input_size=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)


if __name__ == "__main__":
    model = TetrisNet()
    dummy = torch.zeros(1, 4)
    out = model(dummy)
    print(f"Input:  {dummy.shape}")
    print(f"Output: {out.shape} = {out.item():.4f}")
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")
    print("model.py OK")
"""
model.py — Neural Network for Tetris (Board Evaluation)

Architecture: 3-layer fully connected network
    Input:  31 board features (heights, holes, bumpiness, max_h, lines_cleared)
    Hidden: 64 → 64 neurons with ReLU
    Output: 1 scalar — predicted board quality (V-value)

The agent evaluates each possible placement by:
    1. Simulating the placement
    2. Extracting board features from the result
    3. Feeding features through this network → quality score
    4. Picking the placement with the highest score
"""

import torch
import torch.nn as nn


class TetrisNet(nn.Module):
    """
    Board evaluation network for Tetris.

    Input  : board feature vector (31 floats)
    Output : single scalar — predicted board quality
    """

    def __init__(self, input_size: int = 31):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),

            nn.Linear(64, 64),
            nn.ReLU(),

            nn.Linear(64, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


if __name__ == "__main__":
    model  = TetrisNet()
    dummy  = torch.zeros(1, 31)
    output = model(dummy)
    print(f"Input shape  : {dummy.shape}")
    print(f"Output shape : {output.shape}")    # should be (1, 1)
    print(f"Score        : {output.item():.4f}")
    print(f"Parameters   : {sum(p.numel() for p in model.parameters()):,}")
    print("model.py sanity check passed.")
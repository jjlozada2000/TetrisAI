import torch
import torch.nn as nn


class TetrisNet(nn.Module):
    def __init__(self, input_size: int = 45, n_actions: int = 6):
        super().__init__()

        self.net = nn.Sequential(
            # Layer 1
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Dropout(p=0.1),

            # Layer 2
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(p=0.1),

            # Layer 3
            nn.Linear(256, 128),
            nn.ReLU(),

            # Output — one Q-value per action
            nn.Linear(128, n_actions),
        )

        # Initialise weights with He initialisation (best practice for ReLU nets)
        self._init_weights()

    def _init_weights(self):
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


if __name__ == "__main__":
    # Quick sanity check
    model = TetrisNet()
    dummy  = torch.zeros(1, 45)
    output = model(dummy)
    print(f"Input shape  : {dummy.shape}")
    print(f"Output shape : {output.shape}")
    print(f"Q-values     : {output.detach().numpy()}")
    print(f"Parameters   : {sum(p.numel() for p in model.parameters()):,}")
    print("model.py sanity check passed.")
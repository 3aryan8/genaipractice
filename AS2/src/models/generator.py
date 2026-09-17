import torch
from torch import nn


class Generator(nn.Module):

    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int = 256,
    ):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),

            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),

            nn.Linear(hidden_dim * 2, 28 * 28),
            nn.Tanh(),
        )

    def forward(self, noise: torch.Tensor) -> torch.Tensor:
        generated = self.model(noise)

        return generated.view(
            noise.size(0),
            1,
            28,
            28,
        )

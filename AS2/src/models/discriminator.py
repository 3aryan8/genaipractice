import torch
from torch import nn


class Discriminator(nn.Module):

    def __init__(
        self,
        hidden_dim: int = 256,
    ):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(28 * 28, hidden_dim),
            nn.LeakyReLU(0.2),

            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(0.2),

            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        flattened = images.view(images.size(0), -1)

        return self.model(flattened)

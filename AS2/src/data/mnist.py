from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from utils.config import config


def create_mnist_dataloader(
    batch_size: int,
    shuffle: bool = True,
) -> DataLoader:
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])

    dataset = datasets.MNIST(
        root=config.raw_dataset,
        train=True,
        download=False,
        transform=transform,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
    )

from pathlib import Path

from torchvision import datasets

from utils.config import config as central_config


class InstallDataset:

    @staticmethod
    def get_mnist(raw_dir: Path) -> Path:
        data_dir = Path(raw_dir) / "MNIST"
        dataset_raw_dir = data_dir / "raw"

        required_files = [
            "train-images-idx3-ubyte",
            "train-labels-idx1-ubyte",
            "t10k-images-idx3-ubyte",
            "t10k-labels-idx1-ubyte",
        ]

        downloaded = all(
            (dataset_raw_dir / file).exists()
            for file in required_files
        )

        if not downloaded:
            print("MNIST NOT FOUND. DOWNLOADING...")

            datasets.MNIST(
                root=data_dir,
                train=True,
                download=True,
            )

            datasets.MNIST(
                root=data_dir,
                train=False,
                download=True,
            )

        else:
            print("MNIST ALREADY DOWNLOADED.")

        return data_dir


class Config:

    root = central_config.root

    as_root = root / "AS2"

    data = as_root / "data"
    raw_dir = data / "raw"

    raw_dataset = InstallDataset.get_mnist(raw_dir)


config = Config()

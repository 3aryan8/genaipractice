from pathlib import Path


class CentralConfig:
    root = Path(__file__).resolve().parent.parent


config = CentralConfig()

import torch


class ExperimentConfig:

    # Data
    batch_size = 64

    # Model
    latent_dim = 100

    # Training
    epochs = 50

    # Standard GAN
    gan_learning_rate = 0.0002

    # WGAN
    wgan_learning_rate = 0.00005
    critic_steps = 5
    clip_value = 0.01

    # Runtime
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )


config = ExperimentConfig()

import torch

from data.mnist import create_mnist_dataloader
from models.critic import Critic
from models.discriminator import Discriminator
from models.generator import Generator
from trainers.gan_trainer import train_gan
from trainers.wgan_trainer import train_wgan
from utils.experiment_config import config


def main():
    print(f"Using device: {config.device}")

    dataloader = create_mnist_dataloader(
        batch_size=config.batch_size,
        shuffle=True,
    )

    # =========================
    # Standard GAN
    # =========================

    print("\nStarting Standard GAN...")

    gan_generator = Generator(
        latent_dim=config.latent_dim,
    )

    discriminator = Discriminator()

    gan_generator, discriminator, gan_history = train_gan(
        generator=gan_generator,
        discriminator=discriminator,
        dataloader=dataloader,
        latent_dim=config.latent_dim,
        epochs=config.epochs,
        device=config.device,
        learning_rate=config.gan_learning_rate,
    )

    # =========================
    # WGAN
    # =========================

    print("\nStarting WGAN...")

    wgan_generator = Generator(
        latent_dim=config.latent_dim,
    )

    critic = Critic()

    wgan_generator, critic, wgan_history = train_wgan(
        generator=wgan_generator,
        critic=critic,
        dataloader=dataloader,
        latent_dim=config.latent_dim,
        epochs=config.epochs,
        device=config.device,
        learning_rate=config.wgan_learning_rate,
        critic_steps=config.critic_steps,
        clip_value=config.clip_value,
    )

    return {
        "gan": {
            "generator": gan_generator,
            "discriminator": discriminator,
            "history": gan_history,
        },
        "wgan": {
            "generator": wgan_generator,
            "critic": critic,
            "history": wgan_history,
        },
    }


if __name__ == "__main__":
    main()

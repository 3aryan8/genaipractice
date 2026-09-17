import torch
from torch import nn, optim


def train_wgan(
    generator: nn.Module,
    critic: nn.Module,
    dataloader,
    latent_dim: int,
    epochs: int,
    device: torch.device,
    learning_rate: float = 0.00005,
    critic_steps: int = 5,
    clip_value: float = 0.01,
):
    generator_optimizer = optim.RMSprop(
        generator.parameters(),
        lr=learning_rate,
    )

    critic_optimizer = optim.RMSprop(
        critic.parameters(),
        lr=learning_rate,
    )

    generator_losses = []
    critic_losses = []

    generator.to(device)
    critic.to(device)

    for epoch in range(epochs):

        epoch_generator_loss = 0.0
        epoch_critic_loss = 0.0
        generator_updates = 0
        critic_updates = 0

        for real_images, _ in dataloader:

            real_images = real_images.to(device)
            batch_size = real_images.size(0)

            # -------------------------
            # Train Critic
            # -------------------------

            for _ in range(critic_steps):

                noise = torch.randn(
                    batch_size,
                    latent_dim,
                    device=device,
                )

                fake_images = generator(noise)

                real_scores = critic(real_images)

                fake_scores = critic(
                    fake_images.detach()
                )

                critic_loss = (
                    fake_scores.mean()
                    - real_scores.mean()
                )

                critic_optimizer.zero_grad()
                critic_loss.backward()
                critic_optimizer.step()

                # Weight clipping
                for parameter in critic.parameters():
                    parameter.data.clamp_(
                        -clip_value,
                        clip_value,
                    )

                epoch_critic_loss += critic_loss.item()
                critic_updates += 1

            # -------------------------
            # Train Generator
            # -------------------------

            noise = torch.randn(
                batch_size,
                latent_dim,
                device=device,
            )

            fake_images = generator(noise)

            fake_scores = critic(fake_images)

            generator_loss = -fake_scores.mean()

            generator_optimizer.zero_grad()
            generator_loss.backward()
            generator_optimizer.step()

            epoch_generator_loss += generator_loss.item()
            generator_updates += 1

        epoch_critic_loss /= critic_updates
        epoch_generator_loss /= generator_updates

        critic_losses.append(epoch_critic_loss)
        generator_losses.append(epoch_generator_loss)

        print(
            f"Epoch [{epoch + 1}/{epochs}] "
            f"C Loss: {epoch_critic_loss:.4f} "
            f"G Loss: {epoch_generator_loss:.4f}"
        )

    history = {
        "generator_loss": generator_losses,
        "critic_loss": critic_losses,
    }

    return generator, critic, history

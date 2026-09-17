import torch
from torch import nn, optim


def train_gan(
    generator: nn.Module,
    discriminator: nn.Module,
    dataloader,
    latent_dim: int,
    epochs: int,
    device: torch.device,
    learning_rate: float = 0.0002,
):
    criterion = nn.BCELoss()

    generator_optimizer = optim.Adam(
        generator.parameters(),
        lr=learning_rate,
        betas=(0.5, 0.999),
    )

    discriminator_optimizer = optim.Adam(
        discriminator.parameters(),
        lr=learning_rate,
        betas=(0.5, 0.999),
    )

    generator_losses = []
    discriminator_losses = []

    generator.to(device)
    discriminator.to(device)

    for epoch in range(epochs):

        epoch_generator_loss = 0.0
        epoch_discriminator_loss = 0.0

        for real_images, _ in dataloader:

            real_images = real_images.to(device)
            batch_size = real_images.size(0)

            real_labels = torch.ones(
                batch_size,
                1,
                device=device,
            )

            fake_labels = torch.zeros(
                batch_size,
                1,
                device=device,
            )

            # -------------------------
            # Train Discriminator
            # -------------------------

            noise = torch.randn(
                batch_size,
                latent_dim,
                device=device,
            )

            fake_images = generator(noise)

            real_predictions = discriminator(real_images)

            fake_predictions = discriminator(
                fake_images.detach()
            )

            real_loss = criterion(
                real_predictions,
                real_labels,
            )

            fake_loss = criterion(
                fake_predictions,
                fake_labels,
            )

            discriminator_loss = real_loss + fake_loss

            discriminator_optimizer.zero_grad()
            discriminator_loss.backward()
            discriminator_optimizer.step()

            # -------------------------
            # Train Generator
            # -------------------------

            noise = torch.randn(
                batch_size,
                latent_dim,
                device=device,
            )

            fake_images = generator(noise)

            fake_predictions = discriminator(fake_images)

            generator_loss = criterion(
                fake_predictions,
                real_labels,
            )

            generator_optimizer.zero_grad()
            generator_loss.backward()
            generator_optimizer.step()

            epoch_discriminator_loss += discriminator_loss.item()
            epoch_generator_loss += generator_loss.item()

        num_batches = len(dataloader)

        epoch_discriminator_loss /= num_batches
        epoch_generator_loss /= num_batches

        discriminator_losses.append(epoch_discriminator_loss)
        generator_losses.append(epoch_generator_loss)

        print(
            f"Epoch [{epoch + 1}/{epochs}] "
            f"D Loss: {epoch_discriminator_loss:.4f} "
            f"G Loss: {epoch_generator_loss:.4f}"
        )

    history = {
        "generator_loss": generator_losses,
        "discriminator_loss": discriminator_losses,
    }

    return generator, discriminator, history

from collections import OrderedDict
from typing import List

import hydra
import pytorch_lightning as pl
import torch
import torchvision
import torchvision.transforms as transforms
import wandb
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

import loss_regularizers.base
import losses.base
import models
import utils


class MsgGAN(pl.LightningModule):
    generator: torch.nn.Module
    discriminator: torch.nn.Module
    train_dataset: torch.utils.data.Dataset
    loss: losses.base.Loss
    discriminator_loss_regularizers: List[loss_regularizers.base.LossRegularizer]
    generator_loss_regularizers: List[loss_regularizers.base.LossRegularizer]
    real_images: torch.Tensor

    def __init__(self, hparams=None):
        super().__init__()
        self.hparams = hparams
        self.cfg = OmegaConf.create(self.hparams)

    def forward(self, z: torch.Tensor):
        return self.generator(z)

    def setup(self, step):
        self.generator = models.MsgGenerator(
            self.cfg.model.generator.filter_multiplier,
            self.cfg.model.generator.min_filters,
            self.cfg.model.generator.max_filters,
            self.cfg.image_size,
            self.cfg.dataset.image_channels,
            self.cfg.latent_dimension,
            self.cfg.spectral_normalization
        )

        self.discriminator = models.MsgDiscriminator(
            self.cfg.model.generator.filter_multiplier,
            self.cfg.model.generator.min_filters,
            self.cfg.model.generator.max_filters,
            self.cfg.image_size,
            self.cfg.dataset.image_channels,
            self.cfg.spectral_normalization
        )

        self.loss = hydra.utils.instantiate(self.cfg.loss)

        self.discriminator_loss_regularizers = [
            hydra.utils.instantiate(regularizer)
            for regularizer in self.cfg.loss_regularizers.discriminator
        ]

        self.generator_loss_regularizers = [
            hydra.utils.instantiate(regularizer)
            for regularizer in self.cfg.loss_regularizers.generator
        ]

    def training_step(self, batch, batch_idx, optimizer_idx):
        self.discriminator.train(optimizer_idx == 0)
        self.generator.train(optimizer_idx == 1)

        if optimizer_idx == 0:  # Train discriminator
            return self.training_step_discriminator(batch)

        if optimizer_idx == 1:  # Train generator
            return self.training_step_generator(batch)

    def training_step_discriminator(self, batch):
        self.real_images, _ = batch

        z = utils.sample_noise(self.real_images.size(0), self.cfg.latent_dimension, self.real_images.device)
        scaled_real_images = utils.to_scaled_images(self.real_images, self.cfg.image_size)
        fake_images = [fake_image.detach() for fake_image in self.forward(z)]

        real_validity = self.discriminator(scaled_real_images)
        fake_validity = self.discriminator(fake_images)

        loss = self.loss.discriminator_loss(real_validity, fake_validity)

        regularizers = {
            r.log_as: r(self.discriminator, scaled_real_images, fake_images)
            for r
            in self.discriminator_loss_regularizers
        }

        logs = {
            "d_loss": loss,
            **regularizers
        }

        return OrderedDict({"loss": loss + sum(regularizers.values()), "log": logs, "progress_bar": logs})

    def training_step_generator(self, batch):
        self.real_images, _ = batch

        z = utils.sample_noise(self.real_images.size(0), self.cfg.latent_dimension, self.real_images.device)

        scaled_real_images = utils.to_scaled_images(self.real_images, self.cfg.image_size)
        fake_images = self.forward(z)

        real_validity = self.discriminator(scaled_real_images)
        fake_validity = self.discriminator(fake_images)

        loss = self.loss.generator_loss(real_validity, fake_validity)

        regularizers = {
            r.log_as: r(self.generator, scaled_real_images, fake_images)
            for r
            in self.generator_loss_regularizers
        }

        logs = {
            "g_loss": loss,
            **regularizers
        }
        return OrderedDict({"loss": loss + sum(regularizers.values()), "log": logs, "progress_bar": logs})

    def prepare_data(self) -> None:
        transform = transforms.Compose([
            transforms.Resize((self.cfg.image_size, self.cfg.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

        self.train_dataset = hydra.utils.instantiate(
            self.cfg.dataset,
            hydra.utils.to_absolute_path(self.cfg.datasets_path),
            train=True,
            download=True
        )

        self.train_dataset.train = True
        self.train_dataset.download = True
        self.train_dataset.transform = transform

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        return DataLoader(
            batch_size=self.cfg.dataloader.batch_size,
            drop_last=self.cfg.dataloader.drop_last,
            num_workers=self.cfg.dataloader.num_workers,
            dataset=self.train_dataset
        )

    def configure_optimizers(self):
        discriminator_optimizer = hydra.utils.instantiate(self.cfg.optimizer.discriminator, self.discriminator.parameters())
        generator_optimizer = hydra.utils.instantiate(self.cfg.optimizer.generator, self.generator.parameters())

        return (
            {"optimizer": discriminator_optimizer, "frequency": self.cfg.optimizer.discriminator.frequency},
            {"optimizer": generator_optimizer, "frequency": self.cfg.optimizer.generator.frequency}
        )

    def on_epoch_end(self) -> None:
        z = utils.sample_noise(self.cfg.logging.image_grid_size ** 2, self.cfg.latent_dimension, self.real_images.device)
        resolutions = self.forward(z)

        for resolution in resolutions:
            grid = wandb.Image(
                torchvision.utils.make_grid(resolution.detach(), nrow=self.cfg.logging.image_grid_size, padding=1)
            )

            self.logger.experiment.log({
                str(resolution.size(2)) + "x" + str(resolution.size(3)): grid
            }, step=self.global_step)

        name = " ".join(str(self.cfg.name).split()).replace(" ", "-").lower()
        path = f"{hydra.utils.to_absolute_path(self.cfg.checkpoints_path)}/{name}--{self.logger.version}.pth"
        self.trainer.save_checkpoint(path)

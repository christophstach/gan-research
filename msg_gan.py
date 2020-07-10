from collections import OrderedDict
from typing import List, Dict, Any

import hydra
import pytorch_lightning as pl
import torch
import torchvision
import torchvision.transforms as transforms
import wandb
from omegaconf import OmegaConf
from torch import Tensor
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader

import gradient_regularizers.base
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
    discriminator_gradient_regularizers: List[gradient_regularizers.base.GradientRegularizer]
    generator_loss_regularizers: List[loss_regularizers.base.LossRegularizer]
    generator_gradient_regularizers: List[gradient_regularizers.base.GradientRegularizer]
    real_images: torch.Tensor

    def __init__(self, hparams=None):
        super().__init__()
        self.hparams = hparams
        self.cfg = OmegaConf.create(self.hparams)

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
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

        self.discriminator_gradient_regularizers = [
            hydra.utils.instantiate(regularizer)
            for regularizer in self.cfg.gradient_regularizers.discriminator
        ]

        self.generator_loss_regularizers = [
            hydra.utils.instantiate(regularizer)
            for regularizer in self.cfg.loss_regularizers.generator
        ]

        self.generator_gradient_regularizers = [
            hydra.utils.instantiate(regularizer)
            for regularizer in self.cfg.gradient_regularizers.generator
        ]

    def training_step(self, batch, batch_idx, optimizer_idx):
        self.discriminator.train(optimizer_idx == 0)
        self.generator.train(optimizer_idx == 1)

        if optimizer_idx == 0: return self.training_step_discriminator(batch)
        if optimizer_idx == 1: return self.training_step_generator(batch)

    def training_step_discriminator(self, batch):
        self.real_images, _ = batch
        logs = {}

        z = utils.sample_noise(self.real_images.size(0), self.cfg.latent_dimension, self.real_images.device)
        scaled_real_images = utils.to_scaled_images(self.real_images, self.cfg.image_size)
        fake_images = [fake_image.detach() for fake_image in self.forward(z)]

        if self.cfg.instance_noise:
            scaled_real_images, _ = utils.instance_noise(scaled_real_images, self.global_step, self.cfg.instance_noise_last_global_step)
            fake_images, in_sigma = utils.instance_noise(fake_images, self.global_step, self.cfg.instance_noise_last_global_step)
            logs = {**logs, "in_sigma": in_sigma}

        real_validity = self.discriminator(scaled_real_images)
        fake_validity = self.discriminator(fake_images)

        loss = self.loss.discriminator_loss(real_validity, fake_validity)

        regularizers = {
            r.log_as: r(self.discriminator, scaled_real_images, fake_images)
            for r
            in self.discriminator_loss_regularizers
        }

        logs = {
            **logs,
            **regularizers,
            "d_loss": loss
        }

        return OrderedDict({
            "loss": loss + sum(regularizers.values()),
            "log": logs,
            "progress_bar": logs
        })

    def training_step_generator(self, batch):
        self.real_images, _ = batch
        logs = {}

        z = utils.sample_noise(self.real_images.size(0), self.cfg.latent_dimension, self.real_images.device)
        scaled_real_images = utils.to_scaled_images(self.real_images, self.cfg.image_size)
        fake_images = self.forward(z)

        if self.cfg.instance_noise:
            scaled_real_images, _ = utils.instance_noise(scaled_real_images, self.global_step, self.cfg.instance_noise_last_global_step)
            fake_images, in_sigma = utils.instance_noise(fake_images, self.global_step, self.cfg.instance_noise_last_global_step)
            logs = {**logs, "in_sigma": in_sigma}

        real_validity = self.discriminator(scaled_real_images)
        fake_validity = self.discriminator(fake_images)

        loss = self.loss.generator_loss(real_validity, fake_validity)

        regularizers = {
            r.log_as: r(self.generator, scaled_real_images, fake_images)
            for r
            in self.generator_loss_regularizers
        }

        logs = {
            **logs,
            **regularizers,
            "g_loss": loss
        }

        return OrderedDict({
            "loss": loss + sum(regularizers.values()),
            "log": logs,
            "progress_bar": logs
        })

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        transform = transforms.Compose([
            transforms.Resize((self.cfg.image_size, self.cfg.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

        self.train_dataset = hydra.utils.instantiate(
            self.cfg.dataset,
            self.cfg.datasets_path,
            train=True,
            download=True
        )

        self.train_dataset.train = True
        self.train_dataset.download = True
        self.train_dataset.transform = transform

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

    def backward(self, trainer, loss: Tensor, optimizer: Optimizer, optimizer_idx: int) -> None:
        loss.backward()

        if optimizer_idx == 0:
            for discriminator_gradient_regularizer in self.discriminator_gradient_regularizers:
                discriminator_gradient_regularizer(self.discriminator)

        if optimizer_idx == 1:
            for generator_gradient_regularizer in self.generator_gradient_regularizers:
                generator_gradient_regularizer(self.generator)

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
        path = f"{self.cfg.checkpoints_path}/{name}--{self.logger.version}.pth"
        self.trainer.save_checkpoint(path)

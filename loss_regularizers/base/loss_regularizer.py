import abc

import pytorch_lightning as pl
import torch


class LossRegularizer(abc.ABC):
    @abc.abstractmethod
    def __call__(self, model: pl.LightningModule, real_images: torch.Tensor, fake_images: torch.Tensor):
        raise NotImplementedError()

import abc
from typing import List

import torch


class LossRegularizer(abc.ABC):
    log_as: str

    @abc.abstractmethod
    def __call__(self, model: torch.nn.Module, real_images: List[torch.Tensor], fake_images: List[torch.Tensor]):
        raise NotImplementedError()

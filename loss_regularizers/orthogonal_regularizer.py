from typing import List

import torch

import loss_regularizers.base


class OrthogonalRegularizer(loss_regularizers.base.LossRegularizer):
    def __init__(self, log_as, regularization) -> None:
        super().__init__()

        self.log_as = log_as
        self.regularization = regularization

    def __call__(self, model: torch.nn.Module, real_images: List[torch.Tensor], fake_images: List[torch.Tensor]):
        with torch.enable_grad():
            orthogonal_loss = torch.zeros(1, device=real_images[-1].device)

            for name, param in model.named_parameters():
                if 'bias' not in name:
                    param_flat = param.view(param.shape[0], -1)
                    sym = torch.mm(param_flat, torch.t(param_flat))
                    sym -= torch.eye(param_flat.shape[0], device=real_images[-1].device)
                    orthogonal_loss = orthogonal_loss + (self.regularization * sym.abs().sum())

        return orthogonal_loss

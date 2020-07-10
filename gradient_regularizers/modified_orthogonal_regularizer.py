import pytorch_lightning as pl
import torch

import gradient_regularizers.base


# Apply modified ortho reg to a model
# This function is an optimized version that directly computes the gradient,
# instead of computing and then differentiating the loss.
class ModifiedOrthogonalRegularizer(gradient_regularizers.base.GradientRegularizer):
    def __init__(self, regularization, blacklist=None) -> None:
        super().__init__()

        if blacklist is None: blacklist = list()
        self.regularization = regularization
        self.blacklist = blacklist

    def __call__(self, model: pl.LightningModule):
        with torch.no_grad():
            for param in model.parameters():
                # Only apply this to parameters with at least 2 axes, and not in the blacklist
                if len(param.shape) < 2 or any([param is item for item in self.blacklist]): continue

                w = param.view(param.size(0), -1)
                grad = (2 * torch.mm(torch.mm(w, w.t()) * (1. - torch.eye(w.shape[0], device=w.device)), w))
                param.grad.data += self.regularization * grad.view(param.shape)

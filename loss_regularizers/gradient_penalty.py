import torch

import loss_regularizers.base
import utils

from typing import List


class GradientPenalty(loss_regularizers.base.LossRegularizer):
    def __init__(self, log_as, center, coefficient, power, norm_type, penalty_type) -> None:
        super().__init__()

        self.log_as = log_as
        self.center = center
        self.coefficient = coefficient
        self.power = power
        self.norm_type = norm_type
        self.penalty_type = penalty_type

    def __call__(self, model: torch.nn.Module, real_images: List[torch.Tensor], fake_images: List[torch.Tensor]):
        real_images = real_images[-1]
        fake_images = fake_images[-1]

        alpha = torch.rand(real_images.size(0), 1, 1, 1, device=real_images.device)

        interpolates = alpha * real_images + (1 - alpha) * fake_images
        interpolates.requires_grad_()

        interpolates = utils.to_scaled_images(interpolates, real_images.size(-1))

        interpolates_validity = model(interpolates)
        inputs_gradients = torch.autograd.grad(
            outputs=interpolates_validity,
            inputs=interpolates,
            grad_outputs=torch.ones_like(interpolates_validity, device=real_images.device),
            create_graph=True
        )

        inputs_gradients = [input_gradients.view(input_gradients.size(0), -1) for input_gradients in inputs_gradients]
        gradients = torch.cat(inputs_gradients, dim=1)

        if self.norm_type == "l1":
            gradients_norm = gradients.norm(1, dim=1)
        elif self.norm_type == "l2":
            gradients_norm = gradients.norm(2, dim=1)
        elif self.norm_type == "linf":
            gradients_norm, _ = torch.max(torch.abs(gradients), dim=1)
        else:
            raise NotImplementedError()

        if self.penalty_type == "ls":
            penalties = (gradients_norm - self.center) ** self.power
        elif self.penalty_type == "hinge":
            penalties = torch.relu(gradients_norm - self.center)
        else:
            raise NotImplementedError()

        return self.coefficient * penalties.mean().unsqueeze(0)

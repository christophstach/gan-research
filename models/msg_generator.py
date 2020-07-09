import math
from typing import List

import torch

import layers as l


class MsgGenerator(torch.nn.Module):
    def __init__(self, filter_multiplier, min_filters, max_filters, image_size, image_channels, lantent_dimension) -> None:
        super().__init__()

        self.blocks = torch.nn.ModuleList()
        self.to_rgb_converters = torch.nn.ModuleList()

        generator_filters = [
            2 ** (x + 1) * filter_multiplier
            for x in reversed(range(1, int(math.log2(image_size))))
        ]

        if min_filters > 0:
            generator_filters = [
                filter if filter > min_filters
                else min_filters
                for filter in generator_filters
            ]

        if max_filters > 0:
            generator_filters = [
                filter if filter < max_filters
                else max_filters
                for filter in generator_filters
            ]

        # TODO check this hier
        generator_filters[0] = lantent_dimension

        for i, _ in enumerate(generator_filters):
            if i == 0:
                self.blocks.append(
                    l.MsgGeneratorFirstBlock(
                        generator_filters[i]
                    )
                )
            elif i < len(generator_filters) - 1:
                self.blocks.append(
                    l.MsgGeneratorIntermediateBlock(
                        generator_filters[i - 1],
                        generator_filters[i],
                    )
                )
            else:
                self.blocks.append(
                    l.MsgGeneratorLastBlock(
                        generator_filters[i - 1],
                        generator_filters[i]
                    )
                )

            self.to_rgb_converters.append(
                torch.nn.Conv2d(
                    in_channels=generator_filters[i],
                    out_channels=image_channels,
                    kernel_size=1,
                    stride=1,
                    padding=0
                )
            )

    def forward(self, z) -> List[torch.Tensor]:
        outputs = []
        x = z.view(z.size(0), -1, 1, 1)

        for block, to_rgb in zip(self.blocks, self.to_rgb_converters):
            x = block(x)
            output = torch.tanh(to_rgb(x))
            outputs.append(output)

        return outputs

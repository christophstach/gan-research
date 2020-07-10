import abc
import pytorch_lightning as pl


class GradientRegularizer(abc.ABC):
    @abc.abstractmethod
    def __call__(self, model: pl.LightningModule):
        raise NotImplementedError()

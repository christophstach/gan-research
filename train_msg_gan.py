import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf

from msg_gan import MsgGAN


@hydra.main(config_path="configs/msg_gan/config.yaml")
def train_msg_gan(cfg: DictConfig) -> None:
    pl.seed_everything(cfg.training.seed)
    hparams = OmegaConf.to_container(cfg, resolve=True)

    model_checkpoint = hydra.utils.instantiate(cfg.callbacks.model_checkpoint)
    logger = hydra.utils.instantiate(cfg.logging.logger)
    model = MsgGAN(hparams=hparams)

    trainer = pl.Trainer(
        checkpoint_callback=model_checkpoint,
        logger=logger,
        **cfg.trainer,
    )

    trainer.fit(model)


if __name__ == "__main__":
    train_msg_gan()

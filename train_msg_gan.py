import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf

from msg_gan import MsgGAN


@hydra.main(config_path="configs/msg_gan/config.yaml")
def train_msg_gan(cfg: DictConfig) -> None:
    if cfg.resume_id:
        resume_train_msg_gan(cfg)
    else:
        start_train_msg_gan(cfg)


def start_train_msg_gan(cfg: DictConfig) -> None:
    pl.seed_everything(cfg.training.seed)
    hparams = OmegaConf.to_container(cfg, resolve=True)

    logger = hydra.utils.instantiate(cfg.logging.logger)
    model = MsgGAN(hparams=hparams)

    trainer = pl.Trainer(
        checkpoint_callback=False,
        logger=logger,
        weights_summary=None,
        **cfg.trainer
    )
    trainer.fit(model)


def resume_train_msg_gan(cfg: DictConfig) -> None:
    path = f"{hydra.utils.to_absolute_path(cfg.checkpoints_path)}/{cfg.resume_id}.pth"
    hparams = OmegaConf.to_container(cfg, resolve=True)

    _, version = cfg.resume_id.split("--")

    logger = hydra.utils.instantiate(cfg.logging.logger, version=version)
    model = MsgGAN(hparams)

    trainer = pl.Trainer(
        checkpoint_callback=False,
        logger=logger,
        weights_summary=None,
        **cfg.trainer,
        resume_from_checkpoint=path
    )
    trainer.fit(model)


if __name__ == "__main__":
    train_msg_gan()

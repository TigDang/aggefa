import os

import hydra
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from pytorch_lightning import seed_everything


@hydra.main(version_base=None, config_path="config", config_name="test")
def test(cfg: DictConfig):
    assert os.path.exists(
        cfg.ckpt_path
    ), "You must define ckpt path, e.g. ckpt_path=..."

    # Установка начального seed для воспроизводимости
    seed_everything(cfg.seed)
    torch.set_float32_matmul_precision("high")

    # Инициализация модели, датамодуля и трейнера из конфигурации
    datamodule: pl.LightningDataModule = hydra.utils.instantiate(cfg.datamodule)
    module: pl.LightningModule = hydra.utils.instantiate(cfg.module, _recursive_=False)
    trainer: pl.Trainer = hydra.utils.instantiate(cfg.trainer)

    # Тестировка модели
    trainer.test(module, datamodule, ckpt_path=cfg.ckpt_path)


if __name__ == "__main__":
    test()

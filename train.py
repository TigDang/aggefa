import hydra
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from pytorch_lightning import seed_everything


@hydra.main(version_base=None, config_path="config", config_name="train")
def train(cfg: DictConfig):
    # Установка начального seed для воспроизводимости
    seed_everything(cfg.seed)
    torch.set_float32_matmul_precision("high")

    # Инициализация модели, датамодуля и трейнера из конфигурации
    datamodule = hydra.utils.instantiate(cfg.datamodule)
    module = hydra.utils.instantiate(cfg.module, _recursive_=False)
    trainer = hydra.utils.instantiate(cfg.trainer)

    # Тренировка модели
    trainer.fit(module, datamodule)


if __name__ == "__main__":
    train()

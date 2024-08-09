import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig
from pytorch_lightning import seed_everything

from src.datamodule import FaceDatamodule
from src.module import FaceRecognitionModule


@hydra.main(version_base=None, config_path="config", config_name="config")
def train(cfg: DictConfig):
    # Установка начального seed для воспроизводимости
    seed_everything(cfg.seed)

    # Инициализация модели, датамодуля и трейнера из конфигурации
    model = FaceRecognitionModule(**cfg.model)
    datamodule = FaceDatamodule(**cfg.datamodule)
    trainer = pl.Trainer(**cfg.trainer)

    # Тренировка модели
    trainer.fit(model, datamodule)


if __name__ == "__main__":
    train()

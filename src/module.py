from typing import Optional

import hydra
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from omegaconf import DictConfig
from torchmetrics import Accuracy, MeanAbsoluteError
from torchmetrics.detection.mean_ap import MeanAveragePrecision


class ObjectDetector(pl.LightningModule):
    def __init__(self, cfg: DictConfig, model: nn.Module, loss: nn.Module):
        super().__init__()

        self.cfg = cfg
        self.model = hydra.utils.instantiate(model)
        self.loss = hydra.utils.instantiate(loss)
        self.map_metric = MeanAveragePrecision()

    def configure_optimizers(self):
        # Инициализация оптимизатора через Hydra
        optimizer = hydra.utils.instantiate(
            self.cfg.optimizer, params=self.model.parameters()
        )

        # Инициализация шедулер через Hydra
        scheduler = hydra.utils.instantiate(self.cfg.scheduler, optimizer=optimizer)

        # Возвращаем оба объекта в требуемом формате
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": "epoch",
                "frequency": 1,
            },
        }

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, targets = batch
        train_loss = self.model(images, targets)

        self.log(
            "train_loss",
            train_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return train_loss

    def validation_step(self, batch, batch_idx):
        images, targets = batch
        outputs = self.model(images)

        # Подсчет валидационной потери

        val_loss = self.loss(outputs, targets)

        # Оценка метрики
        self.map_metric.update(outputs, targets)

        self.log("val_loss", val_loss, on_epoch=True, prog_bar=True, logger=True)
        return val_loss

    def on_validation_epoch_end(self, outputs):
        # Расчет Mean Average Precision для всех валидационных данных
        map_score = self.map_metric.compute()
        self.log("val_map", map_score, prog_bar=True, logger=True)
        self.map_metric.reset()

    def test_step(self, batch, batch_idx):
        images, targets = batch
        outputs = self.model(images)

        # Оценка метрики на тестовых данных
        self.map_metric.update(outputs, targets)

        return outputs

    def on_test_epoch_end(self, outputs):
        # Расчет Mean Average Precision для всех тестовых данных
        map_score = self.map_metric.compute()
        self.log("test_map", map_score, prog_bar=True, logger=True)
        self.map_metric.reset()

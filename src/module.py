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
from torchvision import models
from torchvision.models import ResNet18_Weights


class AgeGenderPredictor(pl.LightningModule):
    def __init__(
        self, cfg: DictConfig, gender_metric: DictConfig, age_metric: DictConfig
    ):
        super().__init__()

        self.cfg = cfg
        self.gender_metric = hydra.utils.instantiate(gender_metric)
        self.age_metric = hydra.utils.instantiate(age_metric)

        # Используем предобученную ResNet18 в качестве backbone
        self.backbone = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

        # Замена последнего слоя на два отдельных слоя для возраста и пола
        self.age_head = nn.Linear(num_features, 1)  # Возраст - регрессия
        self.gender_head = nn.Linear(num_features, 1)  # Пол - классификация

    def configure_optimizers(self):
        # Инициализация оптимизатора через Hydra
        optimizer = hydra.utils.instantiate(
            self.cfg.optimizer, params=self.parameters()
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
        features = self.backbone(x)
        age = self.age_head(features)
        gender = self.gender_head(features)
        return age, gender

    def training_step(self, batch, batch_idx):
        images, targets = batch
        age_preds, gender_preds = self(images)
        age_loss = F.mse_loss(age_preds, targets["age"])
        gender_loss = F.cross_entropy(gender_preds, targets["gender"])
        loss = age_loss + gender_loss
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        images, targets = batch
        age_preds, gender_preds = self(images)
        age_loss = F.mse_loss(age_preds, targets["age"])
        self.age_metric.update(age_preds, targets["age"])

        gender_loss = F.cross_entropy(gender_preds, targets["gender"])
        self.gender_metric.update(gender_preds, targets["gender"])
        loss = age_loss + gender_loss
        self.log("val_loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        images, targets = batch
        age_preds, gender_preds = self(images)
        self.age_metric.update(age_preds, targets["age"])
        self.gender_metric.update(gender_preds, targets["gender"])

    def on_validation_epoch_end(self):
        # Расчет Mean Average Precision для всех валидационных данных
        class_score = self.gender_metric.compute()
        reg_score = self.age_metric.compute()

        self.log("gender_metric_val", class_score, prog_bar=True, logger=True)
        self.gender_metric.reset()

        self.log("age_metric_val", reg_score, prog_bar=True, logger=True)
        self.age_metric.reset()

    def on_test_epoch_end(self):
        # Расчет Mean Average Precision для всех тестовых данных
        class_score = self.gender_metric.compute()
        reg_score = self.age_metric.compute()

        self.log("gender_metric_test", class_score, prog_bar=True, logger=True)
        self.gender_metric.reset()

        self.log("age_metric_test", reg_score, prog_bar=True, logger=True)
        self.age_metric.reset()

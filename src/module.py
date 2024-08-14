from typing import Dict, Tuple

import hydra
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig
from torchmetrics import Metric
from torchvision import models
from torchvision.models import ResNet18_Weights


class AgeGenderPredictor(pl.LightningModule):
    def __init__(
        self,
        optimizer: DictConfig,
        scheduler: DictConfig,
        gender_metric: DictConfig,
        age_metric: DictConfig,
    ):
        """Lightning module for training age&gender estimator

        Args:
            optimizer (DictConfig): config for optimizer
            scheduler (DictConfig): config for scheduler
            gender_metric (DictConfig): config for metric which will computed over gender preds (so Classification)
            age_metric (DictConfig): config for metric which will computed over age preds (so Regression)
        """
        super().__init__()

        self.optimizer: DictConfig = optimizer
        self.scheduler: DictConfig = scheduler
        # Инициализация метрик
        self.gender_metric: Metric = hydra.utils.instantiate(gender_metric)
        self.age_metric: Metric = hydra.utils.instantiate(age_metric)

        # Используем предобученную ResNet18 в качестве backbone
        self.backbone: nn.Module = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        num_features: int = self.backbone.fc.in_features

        # Замена последнего слоя на два отдельных слоя для возраста и пола
        # Возраст - регрессия
        self.backbone.fc = nn.Identity()
        self.age_head: nn.Module = nn.Sequential(nn.Linear(num_features, 1))

        # Пол - классификация
        self.gender_head: nn.Module = nn.Sequential(
            nn.Linear(num_features, 1), nn.Sigmoid()
        )

    def configure_optimizers(self) -> Dict:
        # Инициализация оптимизатора через Hydra
        optimizer: torch.optim.optimizer.Optimizer = hydra.utils.instantiate(
            self.optimizer, params=self.parameters()
        )

        # Инициализация шедулер через Hydra
        scheduler: torch.optim.lr_scheduler.LRScheduler = hydra.utils.instantiate(
            self.scheduler, optimizer=optimizer
        )

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

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict age and gender over the batch

        Args:
            x (torch.Tensor): [B, 3, 256, 256]-shaped tensor of input images stacked in batch

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: age and gender preds over batch
        """
        features = self.backbone(x)
        age = self.age_head(features)
        gender = self.gender_head(features)
        return age, gender

    def training_step(
        self, batch: Tuple[torch.Tensor, Dict], batch_idx: int
    ) -> torch.Tensor:
        images, targets = batch
        age_preds, gender_preds = self(images)
        age_loss = F.mse_loss(age_preds, targets["age"])
        gender_loss = F.binary_cross_entropy(gender_preds, targets["gender"])
        loss = age_loss + gender_loss
        self.log("train_loss", loss)
        return loss

    def validation_step(
        self, batch: Tuple[torch.Tensor, Dict], batch_idx: int
    ) -> torch.Tensor:
        images, targets = batch
        age_preds, gender_preds = self(images)
        age_loss = F.mse_loss(age_preds, targets["age"])
        self.age_metric.update(age_preds, targets["age"])

        gender_loss = F.binary_cross_entropy(gender_preds, targets["gender"])
        self.gender_metric.update(gender_preds, targets["gender"])
        loss = age_loss + gender_loss
        self.log("val_loss", loss)
        return loss

    def test_step(self, batch: Tuple[torch.Tensor, Dict], batch_idx: int) -> None:
        images, targets = batch
        age_preds, gender_preds = self(images)
        self.age_metric.update(age_preds, targets["age"])
        self.gender_metric.update(gender_preds, targets["gender"])

    def on_validation_epoch_end(self) -> None:
        # Расчет Mean Average Precision для всех валидационных данных
        class_score = self.gender_metric.compute()
        reg_score = self.age_metric.compute()

        self.log("gender_metric_val", class_score, prog_bar=True, logger=True)
        self.gender_metric.reset()

        self.log("age_metric_val", reg_score, prog_bar=True, logger=True)
        self.age_metric.reset()

    def on_test_epoch_end(self) -> None:
        # Расчет Mean Average Precision для всех тестовых данных
        class_score = self.gender_metric.compute()
        reg_score = self.age_metric.compute()

        self.log("gender_metric_test", class_score, prog_bar=True, logger=True)
        self.gender_metric.reset()

        self.log("age_metric_test", reg_score, prog_bar=True, logger=True)
        self.age_metric.reset()

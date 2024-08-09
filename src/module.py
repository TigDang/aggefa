import hydra
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from torchmetrics import Accuracy, MeanAbsoluteError


class FaceRecognitionModule(pl.LightningModule):
    def __init__(self, model_config: DictConfig):
        super(FaceRecognitionModule, self).__init__()

        # Параметры модели и обучения из Hydra конфига
        self.learning_rate = model_config.learning_rate

        self.face_encoder = self.init_face_encoder(model_config.encoder)
        self.age_predictor = self.init_age_predictor(model_config.age_predictor)
        self.gender_predictor = self.init_gender_predictor(
            model_config.gender_predictor
        )

        # Метрики
        self.age_mae = MeanAbsoluteError()  # MAE для возраста
        self.gender_accuracy = Accuracy()  # Точность для пола

    def init_face_encoder(self, encoder_config: DictConfig):
        # Инициализация face_encoder на основе конфигурации
        return None

    def init_age_predictor(self, age_predictor_config: DictConfig):
        # Инициализация age_predictor на основе конфигурации
        return None

    def init_gender_predictor(self, gender_predictor_config: DictConfig):
        # Инициализация gender_predictor на основе конфигурации
        return None

    def forward(self, x):
        encoded_face = self.face_encoder(x)
        age = self.age_predictor(encoded_face)
        gender = self.gender_predictor(encoded_face)
        return age, gender

    def training_step(self, batch, batch_idx):
        x, y_age, y_gender = batch
        pred_age, pred_gender = self(x)

        # Расчёт потерь
        loss_age = F.mse_loss(pred_age, y_age)  # MSE для возраста
        loss_gender = F.cross_entropy(pred_gender, y_gender)  # CrossEntropy для пола

        # Общая потеря
        loss = loss_age + loss_gender

        # Логирование метрик и потерь
        self.log("train_loss", loss)
        self.log("train_loss_age", loss_age)
        self.log("train_loss_gender", loss_gender)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y_age, y_gender = batch
        pred_age, pred_gender = self(x)

        # Расчёт метрик
        loss_age = F.mse_loss(pred_age, y_age)
        loss_gender = F.cross_entropy(pred_gender, y_gender)

        self.age_mae(pred_age, y_age)
        self.gender_accuracy(pred_gender, y_gender)

        # Логирование метрик и потерь
        self.log("val_loss_age", loss_age)
        self.log("val_loss_gender", loss_gender)
        self.log("val_age_mae", self.age_mae)
        self.log("val_gender_accuracy", self.gender_accuracy)

        return loss_age + loss_gender

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

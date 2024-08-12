import logging
import os
from typing import Callable, Optional

import albumentations as A
import datasets
import hydra
import numpy as np
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from PIL import Image


class UTKFace:
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.image_filenames = os.listdir(image_dir)
        self.transform = transform

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        # Получение имени файла и пути к изображению
        img_name = self.image_filenames[idx]
        img_path = os.path.join(self.image_dir, img_name)

        # Загрузка изображения
        image = Image.open(img_path).convert("RGB")

        # Извлечение возраста и пола из имени файла
        age, gender, _, _ = img_name.split("_")
        age = int(age)
        gender = float(gender)

        if self.transform:
            image = self.transform(image=np.array(image))["image"]

        # Преобразование в тензор
        return image, {
            "age": torch.tensor(age, dtype=torch.float32),
            "gender": torch.tensor(gender, dtype=torch.long),
        }


class FaceDatamodule(pl.LightningDataModule):
    trainset: UTKFace
    valset: UTKFace
    testset: UTKFace

    def __init__(
        self,
        dataset: Callable,
        train_augm: Callable,
        val_augm: Callable,
        test_augm: Callable,
        loaders: DictConfig,
    ):
        super(FaceDatamodule, self).__init__()

        trainlenght = int(len(dataset) * 0.7)
        vallenght = int(len(dataset) * 0.1)
        testlenght = len(dataset) - vallenght - trainlenght

        self.trainset, self.valset, self.testset = torch.utils.data.random_split(
            dataset, [trainlenght, vallenght, testlenght]
        )

        self.trainset.transform = train_augm
        self.valset.transform = val_augm
        self.testset.transform = test_augm

        self.loaders_conf = loaders

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            **self.loaders_conf.train_loader, dataset=self.trainset
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            **self.loaders_conf.val_loader, dataset=self.valset
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            **self.loaders_conf.test_loader, dataset=self.testset
        )

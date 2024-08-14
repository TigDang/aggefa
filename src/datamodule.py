import logging
import os
from typing import Callable, Dict, Optional

import albumentations as A
import datasets
import hydra
import numpy as np
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from PIL import Image


class UTKFace:
    def __init__(
        self, image_dir: os.PathLike, transform: Optional[A.BasicTransform] = None
    ):
        """Dataset of faces with annotation of age and gender

        Args:
            image_dir (os.PathLike): path to folder with images
            transform (Optional[A.BasicTransform], optional): Albumentations transforms. Defaults to None.
        """
        self.image_dir = image_dir
        self.image_filenames = os.listdir(image_dir)
        self.transform = transform

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx) -> Dict[torch.Tensor, torch.Tensor]:
        """
        Returns:
            Dict[torch.Tensor, torch.Tensor]: dictionary with keys 'age' and 'gender' with age and gender tensors
        """

        # Получение имени файла и пути к изображению
        img_name = self.image_filenames[idx]
        img_path = os.path.join(self.image_dir, img_name)

        # Загрузка изображения
        image = Image.open(img_path).convert("RGB")

        # Извлечение возраста и пола из имени файла
        splitted_filename = img_name.split("_")
        age, gender = splitted_filename[0], splitted_filename[1]
        age = float(age)
        gender = float(gender)

        if self.transform:
            image = self.transform(image=np.array(image))["image"]

        # Преобразование в тензор
        return image, {
            "age": torch.tensor([age], dtype=torch.float32),
            "gender": torch.tensor([gender], dtype=torch.float32),
        }


class FaceDatamodule(pl.LightningDataModule):
    trainset: UTKFace
    valset: UTKFace
    testset: UTKFace

    def __init__(
        self,
        dataset: UTKFace,
        train_augm: A.BasicTransform,
        val_augm: A.BasicTransform,
        test_augm: A.BasicTransform,
        loaders: DictConfig,
    ):
        """Datamodule which split UTKFace dataset into the train, validation and test subsets with 0.7, 0.1 and 0.2 parts of dataset.

        Args:
            dataset (UTKFace): dataset to split
            train_augm (A.BasicTransform): albumentations for train
            val_augm (A.BasicTransform):  albumentations for validation
            test_augm (A.BasicTransform):  albumentations for test
            loaders (DictConfig): config of train, val and test dataloaders
        """
        super(FaceDatamodule, self).__init__()

        trainlenght = int(len(dataset) * 0.7)
        vallenght = int(len(dataset) * 0.1)
        testlenght = len(dataset) - vallenght - trainlenght

        self.trainset, self.valset, self.testset = torch.utils.data.random_split(
            dataset, [trainlenght, vallenght, testlenght]
        )

        self.trainset.dataset.transform = train_augm
        self.valset.dataset.transform = val_augm
        self.testset.dataset.transform = test_augm

        self.loaders_conf = loaders

    def setup(self, stage: str):
        return

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

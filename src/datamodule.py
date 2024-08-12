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


class WiderfaceDataset:
    def __init__(
        self,
        split: str,
        transforms: Optional[Callable] = None,
    ):
        dataset = datasets.load_dataset("CUHK-CSE/wider_face", trust_remote_code=True)
        try:
            self.dataset = dataset[split]
        except KeyError:
            logging.error(
                f"There is no such split '{split}'. Available splits is {dataset.keys()}"
            )
        self.transorms = transforms

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        example = self.dataset[idx]
        # Получение изображения
        image = example["image"]
        image = np.array(image)

        # Получение bounding boxes
        invalid_mask = torch.tensor(example["faces"]["invalid"], dtype=torch.bool)
        # Преобразуем список bbox в тензор
        bbox_tensor = torch.tensor(example["faces"]["bbox"], dtype=torch.float32)
        # Выбор валидных боксов
        bboxes = bbox_tensor[~invalid_mask]

        class_labels = ["face"] * len(bboxes)

        try:
            if self.transorms is not None:
                augm_result = self.transorms(
                    image=image, bboxes=bboxes, class_labels=class_labels
                )
                image = augm_result["image"]
                bboxes = augm_result["bboxes"]
                class_labels = augm_result["class_labels"]
        except:
            print()

        bboxes = torch.tensor(bboxes)
        if bboxes.shape != torch.Size([0]):
            x, y, w, h = bboxes.unbind(1)
            bboxes = torch.stack((x, y, x + w, y + h), dim=1).float()
        else:
            bboxes = torch.zeros((1, 4))
        return image, [{"boxes": bboxes, "labels": torch.tensor([1] * bboxes.shape[0])}]


class FaceDatamodule(pl.LightningDataModule):
    def __init__(
        self,
        trainset: Callable,
        valset: Callable,
        testset: Callable,
        loaders: DictConfig,
    ):
        super(FaceDatamodule, self).__init__()
        self.trainset = trainset
        self.valset = valset
        self.testset = testset
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


if __name__ == "__main__":
    from matplotlib import pyplot as plt

    dataset = WiderfaceDataset(split="train")
    plt.imsave("img.jpg", dataset[0][0])
    print()

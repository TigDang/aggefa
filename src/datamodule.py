import logging
from typing import Callable, Optional

import datasets
import hydra
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig


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

    def get_example(self, i):
        example = self.dataset[i]
        # Получение изображения
        image = example["image"]

        # Получение bounding boxes
        bboxes = example["faces"]["bbox"]

        if self.transorms is not None:
            augm_result = self.transorms(image=image, bboxes=bboxes)
            image = augm_result["image"]
            bboxes = augm_result["bboxes"]

        return image, bboxes


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
    dataset = WiderfaceDataset(split="train")
    print()

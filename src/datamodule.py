from typing import Callable, Optional

import datasets
import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig


class WIDERFACEDataset:
    def __init__(
        self,
        split: str,
        transforms: Optional[Callable] = None,
    ):
        dataset = datasets.load_dataset("CUHK-CSE/wider_face", trust_remote_code=True)
        self.dataset = dataset[split]
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
    def __init__(self, datasets_conf: DictConfig, loaders_conf: DictConfig):
        super(FaceDatamodule, self).__init__()

        # Инициализация параметров из Hydra конфигурации
        self.datasets_conf = datasets_conf
        self.loaders_conf = loaders_conf

        self.trainset = hydra.utils.instantiate(datasets_conf.trainset)
        self.valset = hydra.utils.instantiate(datasets_conf.valset)
        self.testset = hydra.utils.instantiate(datasets_conf.testset)

    def setup(self):
        self.train_dataloader()
        self.train_dataloader()
        self.train_dataloader()

    def train_dataloader(self):
        return hydra.utils.instantiate(
            self.loaders.get("train_loader"), dataset=self.trainset
        )

    def val_dataloader(self):
        return hydra.utils.instantiate(
            self.loaders.get("val_loader"), dataset=self.valset
        )

    def test_dataloader(self):
        return hydra.utils.instantiate(
            self.loaders.get("test_loader"), dataset=self.testset
        )


if __name__ == "__main__":
    dataset = WIDERFACEDataset(split="train")
    print()

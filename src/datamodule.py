import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig
from torch.utils.data import DataLoader, random_split


class FaceDatamodule(pl.LightningDataModule):
    def __init__(self, datasets_conf: DictConfig, loaders_conf: DictConfig):
        super(FaceDatamodule, self).__init__()

        # Инициализация параметров из Hydra конфигурации
        self.datasets_conf = datasets_conf
        self.loaders_conf = loaders_conf

        self.trainset = hydra.utils.instantiate(datasets_conf.trainset)
        self.valset = hydra.utils.instantiate(datasets_conf.valset)
        self.testset = hydra.utils.instantiate(datasets_conf.testset)

    # def setup(self):

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

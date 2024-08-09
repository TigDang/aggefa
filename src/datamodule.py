import pytorch_lightning as pl
from omegaconf import DictConfig
from torch.utils.data import DataLoader, random_split


class FaceDataModule(pl.LightningDataModule):
    def __init__(self, dataset, datamodule_config: DictConfig):
        super(FaceDataModule, self).__init__()

        # Инициализация параметров из Hydra конфигурации
        self.dataset = dataset
        self.batch_size = datamodule_config.batch_size
        self.num_workers = datamodule_config.num_workers
        self.val_split = datamodule_config.val_split

    def setup(self, stage=None):
        # Разделение на тренировочный и валидационный наборы
        val_size = int(len(self.dataset) * self.val_split)
        train_size = len(self.dataset) - val_size
        self.train_dataset, self.val_dataset = random_split(
            self.dataset, [train_size, val_size]
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

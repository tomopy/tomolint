import os
import glob
import torch
import lightning as L
import tifffile as tiff
from torchvision import transforms
from skimage import transform
from torch.utils.data import Dataset
import pathlib
import typing
import numpy as np


class TomoClassData(Dataset):
    """
    Custom dataset for tomography images with pytorch

    """

    def __init__(self, data_path, split, dataset_transform=None):
        """
        Args:
            data_path (str): path to the data
            transform (callable, optional): Optional transform to be applied on a sample.
        """

        self.path = data_path
        self.transform = dataset_transform
        self.images = []
        self.labels = {"datasets-with-ring": 0, "datasets-no-ring": 1, "bad-center": 2}
        self._load_data()

    def _load_data(self):
        for subdir in ["bnl", "finfrock", "tomobank"]:
            for label, idx in self.labels.items():
                path = os.path.join(self.path, subdir, label)
                for img in glob.glob(os.path.join(path, "*.tiff")):
                    self.images.append((img, idx))

    def __len__(self):
        return len(self.images)

    ##TODO unique sample
    # def unique_sample(self, idx: int) -> typing.Tuple[np.ndarray, int]:

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_path, label = self.images[idx]
        image = tiff.imread(img_path)
        image = transform.resize(image, (256, 256))

        if self.transform:
            image = self.transform(image)
        return image, label


class LitTomoClassData(L.LightningDataModule):
    """
    PyTorch Lightning DataModule for tomography images

    """

    def __init__(
        self, data_path: pathlib.Path, batch_size: int = 32, num_workers: int = 4
    ):
        """
        Args:
            data_path (str): path to the data
            batch_size (int): batch size
            num_workers (int): number of workers
        """
        super().__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.split = None

        self.transform = transforms.Compose(
            [
                # transforms.RandomHorizontalFlip(),
                # transforms.RandomResizedCrop(
                #     (32, 32), scale=(0.8, 1.0), ratio=(0.9, 1.1)
                # ),
                transforms.ToTensor(),
                # transforms.Normalize(DATA_MEANS, DATA_STD),
            ]
        )

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_dataset = TomoClassData(
                self.data_path, self.split, self.transform
            )
            self.val_dataset = TomoClassData(self.data_path, self.split, self.transform)
        if stage == "test" or stage is None:
            self.test_dataset = TomoClassData(
                self.data_path, self.split, self.transform
            )

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers
        )

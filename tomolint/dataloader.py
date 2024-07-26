# import os
# import glob
# import torch
# import lightning as L
# import tifffile as tiff
# from torchvision import transforms
# from skimage import transform
# from torch.utils.data import Dataset
# import pathlib
# import typing
# import re
# from collections import defaultdict
# import cv2


# class TomoClassData(Dataset):
#     """
#     Custom dataset for tomography images with pytorch

#     """

#     def __init__(self, data_path, split, dataset_transform=None, subset="large"):
#         """
#         Args:
#             data_path (str): path to the data
#             transform (callable, optional): Optional transform to be applied on a sample.
#         """

#         self.path = data_path
#         self.transform = dataset_transform
#         self.subset = subset
#         self.images = []
#         self.labels = {"datasets-with-ring": 0, "datasets-no-ring": 1, "bad-center": 2}
#         self._load_data()

#     def _load_data(self):
#         for subdir in ["bnl", "finfrock", "tomobank"]:
#             for label, idx in self.labels.items():
#                 path = os.path.join(self.path, subdir, label)
#                 img_list = glob.glob(os.path.join(path, "*.tiff"))
#                 subset_size = self._get_subset_size(len(img_list))
#                 self.images.extend([(img, idx) for img in img_list[:subset_size]])

#     def _get_subset_size(self, total_size):
#         if self.subset == "small":
#             return total_size // 1000
#         elif self.subset == "mid":
#             return total_size // 2
#         else:  # 'large'
#             return total_size

#     def __len__(self):
#         return len(self.images)

#     def find_unique_sample(self):
#         unique_samples = defaultdict(set)
#         sample_pattern = re.compile(r"_(.*?)_rec_recon_")
#         for subdir in ["bnl", "finfrock", "tomobank"]:
#             for label in self.labels:
#                 path = os.path.join(self.path, subdir, label)
#                 for img in glob.glob(os.path.join(path, "*.tiff")):
#                     match = sample_pattern.search(img)
#                     if match:
#                         sample_id = match.group(1)
#                         unique_samples[subdir].add(sample_id)
#         return unique_samples

#     def augment_image(self, image):
#         ## add contrast and brightness
#         contrast = 127
#         brightness = 100
#         image = cv2.addWeighted(image, contrast, image, 0, brightness)
#         ## Normalize
#         image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
#         ## rescale
#         image /= 255.0

#         ## TODO: Good to have for training Add more augmentations globally .

#         return image

#     def __getitem__(self, idx):
#         if torch.is_tensor(idx):
#             idx = idx.tolist()
#         img_path, label = self.images[idx]

#         # change to tiff.imread
#         # image = tiff.imread(img_path)
#         # image = transform.resize(image, (256, 256))
#         image = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
#         image = cv2.resize(image, (256, 256))
#         image = self.augment_image(image)

#         if self.transform:
#             image = self.transform(image)
#         return image, label


# class LitTomoClassData(L.LightningDataModule):
#     """
#     PyTorch Lightning DataModule for tomography images

#     """

#     def __init__(
#         self,
#         data_path: pathlib.Path,
#         batch_size: int = 32,
#         num_workers: int = 4,
#         subset="large",
#     ):
#         """
#         Args:
#             data_path (str): path to the data
#             batch_size (int): batch size
#             num_workers (int): number of workers
#         """
#         super().__init__()
#         self.data_path = data_path
#         self.batch_size = batch_size
#         self.num_workers = num_workers
#         self.split = None
#         self.subset = subset

#         self.transform = transforms.Compose(
#             [
#                 # transforms.RandomHorizontalFlip(p=0.5),
#                 # transforms.RandomResizedCrop(size=(224, 224), antialias=True),
#                 transforms.ToTensor(),
#                 # transforms.Normalize(DATA_MEANS, DATA_STD),
#             ]
#         )

#         self.train_dataset = None
#         self.val_dataset = None
#         self.test_dataset = None

#     def setup(self, stage=None):
#         if stage == "fit" or stage is None:
#             self.train_dataset = TomoClassData(
#                 self.data_path, self.split, self.transform, self.subset
#             )
#             self.val_dataset = TomoClassData(
#                 self.data_path, self.split, self.transform, self.subset
#             )
#         if stage == "test" or stage is None:
#             self.test_dataset = TomoClassData(
#                 self.data_path, self.split, self.transform, self.subset
#             )

#     def train_dataloader(self):
#         return torch.utils.data.DataLoader(
#             self.train_dataset,
#             batch_size=self.batch_size,
#             num_workers=self.num_workers,
#             pin_memory=True,
#         )

#     def val_dataloader(self):
#         return torch.utils.data.DataLoader(
#             self.val_dataset,
#             batch_size=self.batch_size,
#             num_workers=self.num_workers,
#             pin_memory=True,
#         )

#     def test_dataloader(self):
#         return torch.utils.data.DataLoader(
#             self.test_dataset,
#             batch_size=self.batch_size,
#             num_workers=self.num_workers,
#             pin_memory=True,
#         )

### New code snippet: Test, Split
import os
import glob
import cv2
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import pytorch_lightning as pl
from collections import defaultdict
import re
import pathlib


class TomoClassData(Dataset):
    """
    Custom dataset for tomography images with pytorch
    """

    def __init__(self, data_path, dataset_transform=None, subset="large", indices=None):
        """
        Args:
            data_path (str): path to the data
            dataset_transform (callable, optional): Optional transform to be applied on a sample.
            subset (str): size of the subset ("small", "mid", "large")
            indices (list, optional): List of indices to use for this dataset split.
        """
        self.path = data_path
        self.transform = dataset_transform
        self.subset = subset
        self.images = []
        self.labels = {"datasets-with-ring": 0, "datasets-no-ring": 1, "bad-center": 2}
        self._load_data()
        if indices is not None:
            self.images = [self.images[i] for i in indices]

    def _load_data(self):
        for subdir in ["bnl", "finfrock", "tomobank"]:
            for label, idx in self.labels.items():
                path = os.path.join(self.path, subdir, label)
                img_list = glob.glob(os.path.join(path, "*.tiff"))
                subset_size = self._get_subset_size(len(img_list))
                self.images.extend([(img, idx) for img in img_list[:subset_size]])

    def _get_subset_size(self, total_size):
        if self.subset == "small":
            return total_size // 1000
        elif self.subset == "mid":
            return total_size // 2
        else:  # 'large'
            return total_size

    def __len__(self):
        return len(self.images)

    def find_unique_sample(self):
        unique_samples = defaultdict(set)
        sample_pattern = re.compile(r"_(.*?)_rec_recon_")
        for subdir in ["bnl", "finfrock", "tomobank"]:
            for label in self.labels:
                path = os.path.join(self.path, subdir, label)
                for img in glob.glob(os.path.join(path, "*.tiff")):
                    match = sample_pattern.search(img)
                    if match:
                        sample_id = match.group(1)
                        unique_samples[subdir].add(sample_id)
        return unique_samples

    def augment_image(self, image):
        # Add contrast and brightness
        contrast = 127
        brightness = 100
        image = cv2.addWeighted(image, contrast, image, 0, brightness)
        # Normalize
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
        # Rescale
        image /= 255.0

        return image

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_path, label = self.images[idx]

        # Read and process the image
        image = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        image = cv2.resize(image, (256, 256))
        image = self.augment_image(image)

        if self.transform:
            image = self.transform(image)
        return image, label


class LitTomoClassData(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for tomography images
    """

    def __init__(
        self,
        data_path: pathlib.Path,
        batch_size: int = 32,
        num_workers: int = 4,
        subset="large",
        val_split=0.2,
        seed=42,
    ):
        """
        Args:
            data_path (str): path to the data
            batch_size (int): batch size
            num_workers (int): number of workers
            subset (str): size of the subset ("small", "mid", "large")
            val_split (float): proportion of the dataset to use for validation
            seed (int): random seed for splitting the data
        """
        super().__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.subset = subset
        self.val_split = val_split
        self.seed = seed

        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage=None):
        full_dataset = TomoClassData(self.data_path, self.transform, self.subset)
        val_size = int(len(full_dataset) * self.val_split)
        train_size = len(full_dataset) - val_size

        self.train_dataset, self.val_dataset = random_split(
            full_dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(self.seed),
        )

        if stage == "test" or stage is None:
            self.test_dataset = TomoClassData(
                self.data_path, self.transform, self.subset
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
        )

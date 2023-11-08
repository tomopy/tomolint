import torch
import pathlib
import typing
import glob
import numpy as np

class MockRingData(torch.utils.data.Dataset):
    """Placeholder `Dataset` for tomography classification datasets."""

    def __init__(
        self,
        num_examples: int = 200,
        num_classes: int = 5,
    ):
        self.labels = torch.randint(
            low=0,
            high=num_classes,
            dtype=torch.uint8,
            size=(num_examples * num_classes,),
        )
        self.images = torch.randint(
            low=0,
            high=256,
            dtype=torch.uint8,
            size=(num_examples * num_classes, 256, 256),
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Add color channel to images
        return (
            self.images[idx, None].repeat(3, 1, 1).to(torch.float),
            self.labels[idx],
        )


class TomoClassData(torch.utils.data.Dataset):
    """Load classification dataset from folder of numpy files.

    Expects files named `*_[class_number].npy` where each file contains only
    examples of the class as a stack.

    """

    def __init__(
        self,
        folder: pathlib.Path,
        split: typing.Tuple[float, float] = (0.0, 1.0),
    ):
        self.labels = []
        self.images = []
        for file in sorted(glob.glob(str(folder / "*_[0-9].npy"))):
            examples = np.load(file)
            lo = int(split[0] * len(examples))
            hi = int(split[1] * len(examples))
            label = int(file.split('_')[-1][:-4])
            self.labels.append(torch.full((hi-lo,), label, dtype=torch.uint8))
            self.images.append(torch.from_numpy(examples[lo:hi]))
        self.labels = torch.concatenate(self.labels, axis=0)
        self.images = torch.concatenate(self.images, axis=0)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Add color channel to images
        return (
            self.images[idx, None].repeat(3, 1, 1).to(torch.float),
            self.labels[idx],
        )

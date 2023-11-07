import torch


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

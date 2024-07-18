import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torchvision
from project.dataset import LitTomographyDataset, TomographyDataset
from torchvision import transforms


def load_image(image_path):
    """
    Load the image
    File Path: data/bnl/with-ring/000000.tif
    """

    img = Image.open(image_path)
    img = np.array(img)
    return img


def visualize_image(dataset, num_images=4):
    """
    Visualize the image
    """
    images = [dataset[idx][0] for idx in range(num_images)]
    orig_images = [Image.fromarray(dataset[idx][0]) for idx in range(num_images)]

    image_np = [np.array(img) for img in images]
    data_arr = np.stack(image_np, axis=0)  # (shape, H, W , C)

    data_mean = (dataset.data / 255.0).mean(axis=(0, 1, 2))
    data_std = (dataset.data / 255.0).std(axis=(0, 1, 2))

    test_transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(data_mean, data_std)]
    )

    orig_images = [test_transform(img) for img in orig_images]

    img_grid = torchvision.utils.make_grid(
        torch.stack(images + orig_images, dim=0), nrow=4, normalize=True, pad_value=0.5
    )
    img_grid = img_grid.permute(1, 2, 0)

    plt.figure(figsize=(8, 8))
    plt.title("reconstructed images")
    plt.imshow(img_grid)
    plt.axis("off")
    plt.show()
    plt.close()


if __name__ == "__main__":
    # img = load_image()
    # plt.imshow(img)
    # plt.show()
    # plt.close()

    file_path = "/data/aabayomi/data"
    # data = LitTomographyDataset(file_path)
    # train_dataloader = data.train_dataloader()
    data = TomographyDataset(file_path)
    print(len(data))
    visualize_image(data)

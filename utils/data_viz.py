import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torchvision
from torchvision import transforms
import tomolint
import pathlib


import matplotlib
from skimage import data, img_as_float
from skimage import exposure


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


# def visual_histogram(self):
#     """
#     Visualize the histogram of the image
#     """
#     img = load_image()
#     plt.hist(img.ravel(), bins=256, range=(0.0, 1.0), fc="k", ec="k")
#     plt.show()
#     plt.close()

matplotlib.rcParams["font.size"] = 8


def plot_img_and_hist(image, axes, bins=256):
    """Plot an image along with its histogram and cumulative histogram."""

    image = img_as_float(image)
    ax_img, ax_hist = axes
    ax_cdf = ax_hist.twinx()

    # Display image
    ax_img.imshow(image, cmap=plt.cm.gray)
    ax_img.set_axis_off()

    # Display histogram
    ax_hist.hist(image.ravel(), bins=bins, histtype="step", color="black")
    ax_hist.ticklabel_format(axis="y", style="scientific", scilimits=(0, 0))
    ax_hist.set_xlabel("Pixel intensity")
    ax_hist.set_xlim(0, 1)
    ax_hist.set_yticks([])

    # Display cumulative distribution
    img_cdf, bins = exposure.cumulative_distribution(image, bins)
    ax_cdf.plot(bins, img_cdf, "r")
    ax_cdf.set_yticks([])

    return ax_img, ax_hist, ax_cdf


def display_img_hist(image_path):
    # Load an example image
    img = load_image(image_path)

    # Gamma
    gamma_corrected = exposure.adjust_gamma(img, 2)

    # Logarithmic
    logarithmic_corrected = exposure.adjust_log(img, 1)

    # Display results
    fig = plt.figure(figsize=(8, 5))
    axes = np.zeros((2, 3), dtype=object)
    axes[0, 0] = plt.subplot(2, 3, 1)
    axes[0, 1] = plt.subplot(2, 3, 2, sharex=axes[0, 0], sharey=axes[0, 0])
    axes[0, 2] = plt.subplot(2, 3, 3, sharex=axes[0, 0], sharey=axes[0, 0])
    axes[1, 0] = plt.subplot(2, 3, 4)
    axes[1, 1] = plt.subplot(2, 3, 5)
    axes[1, 2] = plt.subplot(2, 3, 6)

    ax_img, ax_hist, ax_cdf = plot_img_and_hist(img, axes[:, 0])
    ax_img.set_title("Low contrast image")

    y_min, y_max = ax_hist.get_ylim()
    ax_hist.set_ylabel("Number of pixels")
    ax_hist.set_yticks(np.linspace(0, y_max, 5))

    ax_img, ax_hist, ax_cdf = plot_img_and_hist(gamma_corrected, axes[:, 1])
    ax_img.set_title("Gamma correction")

    ax_img, ax_hist, ax_cdf = plot_img_and_hist(logarithmic_corrected, axes[:, 2])
    ax_img.set_title("Logarithmic correction")

    ax_cdf.set_ylabel("Fraction of total intensity")
    ax_cdf.set_yticks(np.linspace(0, 1, 5))

    # prevent overlap of y-axis labels
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    # img = load_image()
    # plt.imshow(img)
    # plt.show()
    # plt.close()

    # data = LitTomographyDataset(file_path)
    # train_dataloader = data.train_dataloader()

    data = tomolint.LitTomoClassData(
        pathlib.Path("/data/aabayomi/data"), batch_size=4, num_workers=4, subset="small"
    )
    # data.setup()
    print(len(data.train_dataloader()))
    visualize_image(data)
    display_img_hist("data/bnl/with-ring/000000.tif")

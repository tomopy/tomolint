"""
    This script for training the ViT model using the Lightning module.
"""

import tomolint
import pathlib


def train(model_name="vit"):
    data = tomolint.LitTomoClassData(
        pathlib.Path("/data/aabayomi/data"), batch_size=4, num_workers=4, subset="small"
    )
    model, loss, accuracy = tomolint.train_lightning(
        model_name=model_name,
        num_classes=3,
        num_epochs=50,
        batch_size=32,
        datasets=data,
    )


if __name__ == "__main__":
    train("vit")

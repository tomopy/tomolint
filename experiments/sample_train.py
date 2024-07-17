import tomolint
import pathlib
import torch


def train(model_name="cnn"):

    data = tomolint.LitTomoClassData(pathlib.Path("/data/aabayomi/data"))

    model, loss, accuracy = tomolint.train_lightning(
        model_name=model_name,
        num_classes=3,
        num_epochs=20,
        batch_size=32,
        datasets=data,
    )


if __name__ == "__main__":
    train("cnn")

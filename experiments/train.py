import tomolint
import pathlib
import torch
import matplotlib.pyplot as plt
from lightning.pytorch.cli import LightningCLI
from tomolint.cnn import CNNModel
from tomolint.dataloader import TomoClassData
from tomolint.training import RingClassifier
from tomolint.vit import VisionTransformer
import yaml

## TODO : Add the following to the CLI


def cli_main():
    LightningCLI(
        datamodule_class=tomolint.LitTomoClassData,
        model_class=RingClassifier,
        save_config_overwrite=True,
    )


if __name__ == "__main__":
    cli_main()

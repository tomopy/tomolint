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
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint

import os


def cli_main():
    CHECKPOINT_PATH = "../saved_models/tomolint"
    model_name = "vit"
    device = (
        torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    )
    LightningCLI(
        datamodule_class=tomolint.LitTomoClassData,
        model_class=tomolint.RingClassifier,
        trainer_defaults={
            "callbacks": [
                ModelCheckpoint(save_weights_only=True, mode="max", monitor="val_acc"),
                LearningRateMonitor("epoch"),
            ],
            "default_root_dir": os.path.join(CHECKPOINT_PATH, f"{model_name}"),
            "accelerator": "gpu" if str(device).startswith("cuda") else "cpu",
            "devices": 3,
            "max_epochs": 1,
            "log_every_n_steps": 8,
            "enable_progress_bar": True,
            "resume_from_checkpoint": os.path.join(
                CHECKPOINT_PATH, f"{model_name}.ckpt"
            ),
        },
    )


if __name__ == "__main__":
    cli_main()

import tomolint
import pathlib
import torch
import matplotlib.pyplot as plt
from lightning.pytorch.cli import LightningCLI

## TODO : Add the following to the CLI

# def test_training_real(model_name="cnn", num_classes=3, num_epochs=20, batch_size=32):

#     data = tomolint.LitTomoClassData(pathlib.Path("/data/aabayomi/data"))

#     model, loss, accuracy = tomolint.train_lightning(
#         model_name=model_name,
#         num_classes=num_classes,
#         num_epochs=num_epochs,
#         batch_size=batch_size,
#         datasets=data,
#     )

#     val_result = trainer.test(model, val_dataloader, verbose=False)
#     test_result = trainer.test(model, test_loader, verbose=False)
#     result = {"test": test_result[0]["test_acc"], "val": val_result[0]["test_acc"]}


# if __name__ == "__main__":
#     cli = LightningCLI(datamodule_class=BoringDataModule)
#     pass
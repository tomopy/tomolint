import tomolint
import pathlib
import torch


def train(model_name="cnn"):
    data = tomolint.LitTomoClassData(
        pathlib.Path("/data/aabayomi/data"), batch_size=32, num_workers=4
    )

    model, loss, accuracy, results = tomolint.train_lightning(
        model_name=model_name,
        num_classes=3,
        num_epochs=20,
        batch_size=32,
        datasets=data,
    )
    # test_result = trainer.test(model, data.test_dataloaders(), verbose=False)
    # result = {"test": test_result[0]["test_acc"], "val": val_result[0]["test_acc"]}

    # print(results)


if __name__ == "__main__":
    train("cnn")

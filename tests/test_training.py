import tomolint
import pathlib
import torch
import matplotlib.pyplot as plt


def test_training_real(model_name="cnn"):
    data = tomolint.LitTomoClassData(pathlib.Path("/data/aabayomi/data"))

    model, loss, accuracy, result = tomolint.train_lightning(
        model_name=model_name,
        num_classes=3,
        num_epochs=20,
        batch_size=32,
        datasets=data,
    )

    with torch.no_grad():
        traced_model = torch.jit.trace(
            model.classifier,
            torch.rand((1, 1, 256, 256)).to(model.device),
        )

    torch.jit.save(traced_model, "real-classification.torch")

    reloaded_model = torch.jit.load("real-classification.torch")

    random_result = reloaded_model(
        torch.rand((1, 1, 256, 256)).to(model.device),
    )
    print(random_result)


def test_loading():
    dataset = tomolint.TomoClassData(
        pathlib.Path("/data/aabayomi/data"),
        (0.0, 0.8),
    )

    # print(dataset.labels.shape)
    print(len(dataset.images))
    print(dataset.labels)

    dataset = tomolint.TomoClassData(
        pathlib.Path("/data/aabayomi/data"),
        (0.8, 1.0),
    )
    # print(dataset.labels.shape)
    # print(dataset.images.shape)
    print(len(dataset.images))
    print(dataset.labels)


if __name__ == "__main__":
    test_training_real("cnn")
    # test_loading()

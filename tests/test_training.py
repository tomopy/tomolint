import tomolint
import pathlib
import torch
import matplotlib.pyplot as plt


# def test_training_mock():
#     tomolint.train_lightning(
#         model_name="cnn",
#         num_classes=3,
#         num_epochs=10,
#         datasets={
#             "train": tomolint.MockRingData(num_examples=int(200 * 0.8)),
#             "val": tomolint.MockRingData(num_examples=int(200 * 0.2)),
#         },
#     )


# def test_training_simulated():
#     model, loss, accuracy = tomolint.train_lightning(
#         num_classes=5,
#         num_epochs=20,
#         batch_size=32,
#         datasets={
#             "train": tomolint.TomoClassData(
#                 pathlib.Path("./Simulation_data"),
#                 (0.0, 0.9),
#             ),
#             "val": tomolint.TomoClassData(
#                 pathlib.Path("./Simulation_data"),
#                 (0.9, 1.0),
#             ),
#         },
#     )

#     traced_model = torch.jit.trace(
#         model.classifier,
#         torch.rand((1, 3, 256, 256)).to(model.device),
#     )

#     torch.jit.save(traced_model, "simulated-classification.torch")

#     reloaded_model = torch.jit.load("simulated-classification.torch")

#     random_result = reloaded_model(
#         torch.rand((1, 3, 256, 256)).to(model.device),
#     )
#     print(random_result)


def test_training_real():

    data = tomolint.TomoClassData(
        tomolint.TomoClassData(pathlib.Path("/data/aabayomi/data"), (0.0, 0.9))
    )
   

    model, loss, accuracy = tomolint.train_lightning(
        model_name="cnn",
        num_classes=3,
        num_epochs=20,
        batch_size=32,
        datasets={
            "train": data.train_dataloader(),
            "val": data.val_dataloader(),
            "test": data.test_dataloader(),
        },
    )

    with torch.no_grad():
        traced_model = torch.jit.trace(
            model.classifier,
            torch.rand((1, 3, 256, 256)).to(model.device),
        )

    torch.jit.save(traced_model, "real-classification.torch")

    reloaded_model = torch.jit.load("real-classification.torch")

    random_result = reloaded_model(
        torch.rand((1, 3, 256, 256)).to(model.device),
    )
    print(random_result)


def test_loading():

    dataset = tomolint.TomoClassData(
        pathlib.Path("/aabayomi/data"),
        (0.0, 0.8),
    )

    # print(dataset.labels.shape)
    print(len(dataset.images))
    print(dataset.labels)

    dataset = tomolint.TomoClassData(
        pathlib.Path("/aabayomi/data"),
        (0.8, 1.0),
    )
    # print(dataset.labels.shape)
    # print(dataset.images.shape)
    print(len(dataset.images))
    print(dataset.labels)


if __name__ == "__main__":
    # test_training_mock()
    # test_training_simulated()
    # test_training_real()
    test_loading()

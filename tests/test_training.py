import tomolint
import pathlib
import torch
import matplotlib.pyplot as plt


def test_training_mock():
    tomolint.train(
        num_classes=5,
        num_epochs=10,
        datasets={
            "train": tomolint.MockRingData(num_examples=int(200 * 0.8)),
            "val": tomolint.MockRingData(num_examples=int(200 * 0.2)),
        },
    )


def test_training_simulated():
    model, loss, accuracy = tomolint.train(
        num_classes=5,
        num_epochs=200,
        batch_size=16,
        datasets={
            "train": tomolint.TomoClassData(
                pathlib.Path("./Simulation_data"),
                (0.0, 0.9),
            ),
            "val": tomolint.TomoClassData(
                pathlib.Path("./Simulation_data"),
                (0.9, 1.0),
            ),
        },
    )

    torch.save(model, 'ring-classification.torch')

    plt.figure()
    plt.plot(loss['train'], '--')
    plt.plot(loss['val'])
    plt.legend(['training', 'validation'])
    plt.ylabel('Objective Loss')
    plt.xlabel('Epoch')
    plt.title('Ring Classification Training')
    plt.savefig('loss.svg')

    plt.figure()
    plt.plot(accuracy['train'], '--')
    plt.plot(accuracy['val'])
    plt.legend(['training', 'validation'])
    plt.ylabel('Classification Accuracy')
    plt.xlabel('Epoch')
    plt.title('Ring Classification Training')
    plt.savefig('accuracy.svg')


def test_loading():
    dataset = tomolint.TomoClassData(
        pathlib.Path("./Simulation_data"),
        (0.0, 0.8),
    )
    print(dataset.labels.shape)
    print(dataset.images.shape)
    print(dataset.labels)
    dataset = tomolint.TomoClassData(
        pathlib.Path("./Simulation_data"),
        (0.8, 1.0),
    )
    print(dataset.labels.shape)
    print(dataset.images.shape)
    print(dataset.labels)


if __name__ == "__main__":
    # test_training_mock()
    test_training_simulated()
    # test_loading()

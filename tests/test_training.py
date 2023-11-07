import tomolint


def test_training():
    tomolint.train(
        num_classes=5,
        num_epochs=10,
    )


if __name__ == "__main__":
    test_training()

import tomolint
import pathlib
import torch
import matplotlib.pyplot as plt


models = ["vit", "cnn"]
results = {}

for model_name in models:
    data = tomolint.LitTomoClassData(
        pathlib.Path("/data/aabayomi/data"), batch_size=4, num_workers=4, subset="small"
    )
    model, loss, accuracy = tomolint.train(
        model_name=model_name,
        num_classes=3,
        num_epochs=50,
        batch_size=32,
        datasets=data,
    )
    results[model_name] = {"loss": loss, "accuracy": accuracy}

# Plotting the results
for model_name in models:
    epochs = range(1, 51)  # Assuming num_epochs is 50

    plt.figure(figsize=(12, 5))

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, results[model_name]["loss"], label=f"{model_name} Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title(f"{model_name} Loss over Epochs")
    plt.legend()

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, results[model_name]["accuracy"], label=f"{model_name} Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title(f"{model_name} Accuracy over Epochs")
    plt.legend()

    plt.suptitle(f"{model_name} Training Results")
    plt.show()

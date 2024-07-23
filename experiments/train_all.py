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
        num_epochs=1,
        batch_size=4,
        datasets=data,
    )
    results[model_name] = {"loss": loss, "accuracy": accuracy}

# Plotting the results
for model_name in models:
    epochs = range(1, 3)

    plt.figure(figsize=(16, 10))

    # Plot training loss
    plt.subplot(2, 2, 1)
    plt.plot(
        epochs, results[model_name]["loss"]["train"], label=f"{model_name} Train Loss"
    )
    plt.xlabel("Epochs")
    plt.ylabel("Training Loss")
    plt.title(f"{model_name} Training Loss over Epochs")
    plt.legend()

    # Plot validation loss
    plt.subplot(2, 2, 2)
    plt.plot(epochs, results[model_name]["loss"]["val"], label=f"{model_name} Val Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Validation Loss")
    plt.title(f"{model_name} Validation Loss over Epochs")
    plt.legend()

    # Plot training accuracy
    plt.subplot(2, 2, 3)
    plt.plot(
        epochs,
        results[model_name]["accuracy"]["train"],
        label=f"{model_name} Train Accuracy",
    )
    plt.xlabel("Epochs")
    plt.ylabel("Training Accuracy")
    plt.title(f"{model_name} Training Accuracy over Epochs")
    plt.legend()

    # Plot validation accuracy
    plt.subplot(2, 2, 4)
    plt.plot(
        epochs,
        results[model_name]["accuracy"]["val"],
        label=f"{model_name} Val Accuracy",
    )
    plt.xlabel("Epochs")
    plt.ylabel("Validation Accuracy")
    plt.title(f"{model_name} Validation Accuracy over Epochs")
    plt.legend()

    plt.suptitle(f"{model_name} Training and Validation Results")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

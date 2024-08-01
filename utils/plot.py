"""
This script is used to plot the training and validation loss and accuracy over epochs.
It takes the path to the directory containing the csv files and plots the training and validation loss and accuracy over epochs.

"""

import csv
import matplotlib.pyplot as plt
import pandas as pd
import glob


def read(path) -> list:
    """
    Read the csv files
    """
    data = []
    for p in path:
        with open(p, "r") as f:
            reader = csv.reader(f)
            for row in reader:
                data.append(row)
    return data


def preprocess(data) -> tuple:
    """
    Preprocess the data

    Args:
        data: list of data

    """
    train_loss = []
    val_loss = []
    train_acc = []
    val_acc = []

    for dat in data:
        print(dat)
        train_loss.append(dat[4])
        val_loss.append(dat[6])
        train_acc.append(dat[3])
        val_acc.append(dat[5])

    train_loss = list(filter(None, train_loss))
    val_loss = list(filter(None, val_loss))
    train_acc = list(filter(None, train_acc))
    val_acc = list(filter(None, val_acc))

    train_loss = [float(i) for i in train_loss[1:]]
    val_loss = [float(i) for i in val_loss[1:]]
    train_acc = [float(i) for i in train_acc[1:]]
    val_acc = [float(i) for i in val_acc[1:]]

    return train_loss, val_loss, train_acc, val_acc


def plot(train_loss, val_loss, train_acc, val_acc) -> None:
    """
    Plot the training and validation loss and accuracy over epochs
    Args:
        train_loss: list of training loss
        val_loss: list of validation loss
        train_acc: list of training accuracy
        val_acc: list of validation accuracy

    """
    epochs = range(1, len(train_loss) + 1)

    num_plots = 2
    title = [
        ("Training and Validation Loss over Epochs", ("Loss", "Epochs")),
        ("Training and Validation Accuracy over Epochs", ("Accuracy", "Epochs")),
    ]

    fig, axs = plt.subplots(1, num_plots, figsize=(10, 5))

    for idx, title in enumerate(title):
        axs[idx].set_title(title[0])
        axs[idx].set_xlabel(title[1][1])
        axs[idx].set_ylabel(title[1][0])

    for ax, interp in zip(axs, [(train_loss, val_loss), (train_acc, val_acc)]):
        ax.plot(epochs, interp[0], label="Train Loss")
        ax.plot(epochs, interp[1], label="Val Loss")
        ax.grid(True)
        plt.legend()

    plt.show()


def main(file_path):
    path = glob.glob(file_path + "/*.csv")
    data = read(path)
    train_loss, val_loss, train_acc, val_acc = preprocess(data)
    print(len(train_loss), len(val_loss))

    # plot(train_loss, val_loss, train_acc, val_acc)


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python script.py <file_path>")
    else:
        file_path = sys.argv[1]
        main(file_path)

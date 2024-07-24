# import csv
# import matplotlib.pyplot as plt
# import pandas as pd

# import glob


# def read(path):
#     data = []
#     for p in path:
#         with open(p, "r") as f:
#             reader = csv.reader(f)
#             for row in reader:
#                 data.append(row)
#     return data


# def preprocess(data):
#     train_loss = []
#     val_loss = []
#     train_acc = []
#     val_acc = []
#     for dat in data:
#         train_loss.append(dat[4])
#         val_loss.append(dat[6])
#         train_acc.append(dat[3])
#         val_acc.append(dat[5])

#     train_loss = list(filter(None, train_loss))
#     val_loss = list(filter(None, val_loss))
#     train_acc = list(filter(None, train_acc))
#     val_acc = list(filter(None, val_acc))

#     train_loss = [float(i) for i in train_loss[1:]]
#     val_loss = [float(i) for i in val_loss[1:]]
#     train_acc = [float(i) for i in train_acc[1:]]
#     val_acc = [float(i) for i in val_acc[1:]]

#     return train_loss, val_loss, train_acc, val_acc


# def plot(train_loss, val_loss, train_acc, val_acc):
#     epochs = range(1, len(train_loss) + 1)

#     plt.figure(figsize=(16, 10))

#     # Plot training loss
#     plt.subplot(2, 2, 1)
#     plt.plot(epochs, train_loss, label="Train Loss")
#     plt.xlabel("Epochs")
#     plt.ylabel("Training Loss")
#     plt.title("Training Loss over Epochs")
#     plt.legend()

#     # Plot validation loss
#     plt.subplot(2, 2, 2)
#     plt.plot(epochs, val_loss, label="Val Loss")
#     plt.xlabel("Epochs")
#     plt.ylabel("Validation Loss")
#     plt.title("Validation Loss over Epochs")
#     plt.legend()

#     # Plot training accuracy
#     plt.subplot(2, 2, 3)
#     plt.plot(epochs, train_acc, label="Train Accuracy")
#     plt.xlabel("Epochs")
#     plt.ylabel("Training Accuracy")
#     plt.title("Training Accuracy over Epochs")
#     plt.legend()

#     # Plot validation accuracy
#     plt.subplot(2, 2, 4)
#     plt.plot(epochs, val_acc, label="Val Accuracy")
#     plt.xlabel("Epochs")
#     plt.ylabel("Validation Accuracy")
#     plt.title("Validation Accuracy over Epochs")
#     plt.legend()

#     plt.suptitle("Training and Validation Results")
#     plt.tight_layout(rect=[0, 0, 1, 0.96])
#     plt.show()


# p = "/Users/abayomi/Desktop/internship-24/tomolint/experiments"  # update this
# path = glob.glob(p + "/*.csv")
# data = read(path)
# # print(data)
# train_loss, val_loss, train_acc, val_acc = preprocess(data)
# # print(train_loss, val_loss, train_acc, val_acc)
# plot(train_loss, val_loss, train_acc, val_acc)


# import csv
# import matplotlib.pyplot as plt
# import pandas as pd
# import glob

# def read(path):
#     data = []
#     for p in path:
#         with open(p, "r") as f:
#             reader = csv.reader(f)
#             for row in reader:
#                 data.append(row)
#     return data

# def preprocess(data):
#     train_loss = []
#     val_loss = []
#     train_acc = []
#     val_acc = []
#     for dat in data:
#         train_loss.append(dat[4])
#         val_loss.append(dat[6])
#         train_acc.append(dat[3])
#         val_acc.append(dat[5])

#     train_loss = list(filter(None, train_loss))
#     val_loss = list(filter(None, val_loss))
#     train_acc = list(filter(None, train_acc))
#     val_acc = list(filter(None, val_acc))

#     train_loss = [float(i) for i in train_loss[1:]]
#     val_loss = [float(i) for i in val_loss[1:]]
#     train_acc = [float(i) for i in train_acc[1:]]
#     val_acc = [float(i) for i in val_acc[1:]]

#     return train_loss, val_loss, train_acc, val_acc

# def plot(train_loss, val_loss, train_acc, val_acc):
#     epochs = range(1, len(train_loss) + 1)

#     plt.figure(figsize=(10, 5))

#     # Plot training and validation loss
#     plt.plot(epochs, train_loss, label="Train Loss")
#     plt.plot(epochs, val_loss, label="Val Loss")
#     plt.xlabel("Epochs")
#     plt.ylabel("Loss")
#     plt.title("Training and Validation Loss over Epochs")
#     plt.legend()
#     plt.show()

#     plt.figure(figsize=(10, 5))

#     # Plot training and validation accuracy
#     plt.plot(epochs, train_acc, label="Train Accuracy")
#     plt.plot(epochs, val_acc, label="Val Accuracy")
#     plt.xlabel("Epochs")
#     plt.ylabel("Accuracy")
#     plt.title("Training and Validation Accuracy over Epochs")
#     plt.legend()
#     plt.show()

# def main(file_path):
#     path = glob.glob(file_path + "/*.csv")
#     data = read(path)
#     train_loss, val_loss, train_acc, val_acc = preprocess(data)
#     plot(train_loss, val_loss, train_acc, val_acc)

# if __name__ == "__main__":
#     import sys
#     if len(sys.argv) != 2:
#         print("Usage: python script.py <file_path>")
#     else:
#         file_path = sys.argv[1]
#         main(file_path)

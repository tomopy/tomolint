import csv
import matplotlib.pyplot as plt
import pandas as pd


csv_file_path = "metrics.csv"
# Now let's read the data from the CSV file and plot it
df = pd.read_csv(csv_file_path)
# print(df)
# Plot the data
# plt.figure(figsize=(12, 5))
# df.plot(x='epoch', y=['train_loss', 'val_loss'], ax=plt.subplot(1, 2, 1))

# print(df["train_loss"])

# plt.title("Loss over Epochs")
# plt.ylabel("Loss")
# plt.xlabel("Epoch")
# plt.show()
epochs = [0.0, 1.0, 2.0, 4.0, 5.0]
train_loss = [0.608739, 0.534520, 0.518802, 0.510390, 0.503764]
val_loss = [91.960670, 95.586555, 48.546387, 37.726360, 39.346184]
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(epochs, train_loss, label="Train Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")

plt.title("CNN Loss over Epochs")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epochs, val_loss, label="Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Val loss ")
plt.legend()

plt.suptitle("CNN Training Results")
plt.show()

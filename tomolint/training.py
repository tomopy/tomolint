import torch
import torchvision
import tqdm

import tomolint.dataloader


def train(
    num_classes: int = 5,
    num_epochs: int = 10,
    batch_size: int = 32,
) -> torch.nn.Module:
    datasets = {
        "train": tomolint.dataloader.MockRingData(num_examples=int(200 * 0.8)),
        "val": tomolint.dataloader.MockRingData(num_examples=int(200 * 0.2)),
    }

    dataloaders = {
        "train": torch.utils.data.DataLoader(
            datasets["train"],
            batch_size=32,
            shuffle=True,
        ),
        "val": torch.utils.data.DataLoader(
            datasets["val"],
            batch_size=32,
            shuffle=False,
        ),
    }

    # Use a pre-trained AlexNet model
    model = torchvision.models.alexnet(pretrained=True)

    # Modify the classifier to have the desired number of output classes
    model.classifier[6] = torch.nn.Linear(4096, num_classes)

    # Feeze the features portion of the model
    for param in model.features.parameters():
        param.requires_grad = False

    # Define a loss function and an torch.optimizer
    criterion = torch.nn.CrossEntropyLoss()
    torch.optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # Train the model
    for epoch in tqdm.trange(num_epochs, desc="Training"):
        for phase in ["train", "val"]:
            if phase == "train":
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                torch.optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    assert inputs.shape
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    # outputs should be (batches, classes)
                    # labels should be  (batches, )
                    loss = criterion(outputs, labels)

                    if phase == "train":
                        loss.backward()
                        torch.optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(datasets[phase])
            epoch_acc = running_corrects.double() / len(datasets[phase])

            print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

    return model

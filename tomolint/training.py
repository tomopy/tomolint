import torch
import torchvision
import tqdm


def train(
    datasets,
    num_classes: int = 5,
    num_epochs: int = 10,
    batch_size: int = 32,
) -> torch.nn.Module:

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

    # Use a pre-trained model
    model = torchvision.models.resnet18(torchvision.models.ResNet18_Weights.DEFAULT)

    # Feeze the features portion of the model
    for param in model.parameters():
        param.requires_grad = False

    # Modify the classifier to have the desired number of output classes
    model.fc = torch.nn.Linear(512, num_classes)

    # Define a loss function and an torch.optimizer
    criterion = torch.nn.CrossEntropyLoss()
    torch.optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

    losses = {'train': list(), 'val': list()}
    accuracies = {'train': list(), 'val': list()}

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

            losses[phase].append(epoch_loss)
            accuracies[phase].append(epoch_acc)

    # TODO: Add precision output. i.e for each class: (TP) / (TP + FP)
    # TODO: Add recall output. i.e for each class: (TP) / (TP + FN)

    return model, losses, accuracies

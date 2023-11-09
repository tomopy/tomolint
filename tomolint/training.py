import torch
import torchvision
import tqdm
import lightning
import pathlib

from lightning.pytorch.loggers import WandbLogger


class RingClassifier(lightning.LightningModule):
    def __init__(self, num_classes: int):
        super().__init__()

        # Use a pre-trained model
        model = torchvision.models.resnet18(torchvision.models.ResNet18_Weights.DEFAULT)

        # Feeze the features portion of the model
        for param in model.parameters():
            param.requires_grad = False

        # Modify the classifier to have the desired number of output classes
        model.fc = torch.nn.Linear(512, num_classes)

        self.classifier = model
        self.criterion = torch.nn.CrossEntropyLoss()

    def forward(self, inputs):
        return self.classifier(inputs)

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self.forward(inputs)
        _, preds = torch.max(outputs, 1)
        # outputs should be (batches, classes)
        # labels should be  (batches, )
        loss = self.criterion(outputs, labels)
        corrects = torch.sum(preds == labels.data)
        self.log("train/loss", loss, on_step=False, on_epoch=True)
        self.log(
            "train/accuracy",
            corrects.double() / len(inputs),
            on_step=False,
            on_epoch=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self.classifier(inputs)
        _, preds = torch.max(outputs, 1)
        # outputs should be (batches, classes)
        # labels should be  (batches, )
        loss = self.criterion(outputs, labels)
        corrects = torch.sum(preds == labels.data)
        self.log("validation/loss", loss, on_step=False, on_epoch=True)
        self.log(
            "validation/accuracy",
            corrects.double() / len(inputs),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=0.0001)


def train_lightning(
    datasets,
    num_classes: int = 5,
    num_epochs: int = 10,
    batch_size: int = 32,
):
    dataloaders = {
        "train": torch.utils.data.DataLoader(
            datasets["train"],
            batch_size=batch_size,
            shuffle=True,
        ),
        "val": torch.utils.data.DataLoader(
            datasets["val"],
            batch_size=batch_size,
            shuffle=False,
        ),
    }

    trainer = lightning.Trainer(
        max_epochs=num_epochs,
        log_every_n_steps=8,
        logger=WandbLogger(project="tomolint", log_model="all"),
    )
    model = RingClassifier(num_classes)
    trainer.fit(
        model,
        dataloaders["train"],
        dataloaders["val"],
    )

    # download checkpoint locally (if not already cached)
    run = trainer.loggers[0].experiment
    artifact = run.use_artifact(
        f"carterbox/tomolint/model-{trainer.loggers[0].experiment.id}:best",
        type="model",
    )
    artifact_dir = artifact.download()

    # load checkpoint
    model = RingClassifier.load_from_checkpoint(
        pathlib.Path(artifact_dir) / "model.ckpt",
        num_classes=num_classes,
    )
    return model, [], []


def train(
    datasets,
    num_classes: int = 5,
    num_epochs: int = 10,
    batch_size: int = 32,
) -> torch.nn.Module:
    dataloaders = {
        "train": torch.utils.data.DataLoader(
            datasets["train"],
            batch_size=batch_size,
            shuffle=True,
        ),
        "val": torch.utils.data.DataLoader(
            datasets["val"],
            batch_size=batch_size,
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

    losses = {"train": list(), "val": list()}
    accuracies = {"train": list(), "val": list()}

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

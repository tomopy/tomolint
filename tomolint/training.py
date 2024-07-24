import os
import torch
import torchvision
import tqdm
import lightning
import pathlib
from tomolint.vit import VisionTransformer
from tomolint.cnn import CNNModel
from tomolint.vit import VisionTransformer
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping


def create_model(model_name, params):
    if model_name == "cnn":
        model = CNNModel()
        return model
    elif model_name == "vit":
        vit_params = params.get("vit_params", {})
        model = VisionTransformer(**vit_params)
        return model
    elif model_name == "resnet":
        model = torchvision.models.resnet18(torchvision.models.ResNet18_Weights.DEFAULT)
        # Feeze the features portion of the model
        for param in model.parameters():
            param.requires_grad = False
            # Modify the classifier to have the desired number of output classes
            model.fc = torch.nn.Linear(512, self.num_classes)
            return model
    else:
        print("model not available or not implemented")


class RingClassifier(lightning.LightningModule):
    def __init__(self, num_classes: int, model_name: str, params: dict):
        super().__init__()
        self.save_hyperparameters()

        # choose the model
        model = self.create_model(model_name, params)

        self.classifier = model
        self.num_classes = num_classes
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer_params = params.get("optimizer_params", {})

    def create_model(self, model_name, params):
        if model_name == "cnn":
            model = CNNModel()
            return model
        elif model_name == "vit":
            vit_params = params.get("vit_params", {})
            model = VisionTransformer(**vit_params)
            return model
        elif model_name == "resnet":
            model = torchvision.models.resnet18(
                torchvision.models.ResNet18_Weights.DEFAULT
            )
            # Feeze the features portion of the model
            for param in model.parameters():
                param.requires_grad = False

            # Modify the classifier to have the desired number of output classes
            model.fc = torch.nn.Linear(512, self.num_classes)
            return model
        else:
            print("model not available or not implemented")

    def forward(self, inputs):
        return self.classifier(inputs)

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self.forward(inputs)
        loss = self.criterion(outputs, labels)
        accuracy = (outputs.argmax(dim=1) == labels).float().mean()

        _, preds = torch.max(outputs, 1)
        corrects = torch.sum(preds == labels.data)

        self.log(
            "train_loss",
            loss,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            prog_bar=True,
        )
        self.log(
            "train_acc",
            accuracy,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            prog_bar=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self.classifier(inputs)
        loss = self.criterion(outputs, labels)
        accuracy = (outputs.argmax(dim=1) == labels).float().mean()

        _, preds = torch.max(outputs, 1)
        corrects = torch.sum(preds == labels.data)

        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            prog_bar=True,
        )
        self.log(
            "val_acc",
            accuracy,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        return loss

    def test_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self.classifier(inputs)
        loss = self.criterion(outputs, labels)
        accuracy = (outputs.argmax(dim=1) == labels).float().mean()
        self.log(
            "test_loss",
            loss,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            prog_bar=True,
        )
        self.log(
            "test_acc",
            accuracy,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.optimizer_params["lr"])
        return [optimizer]


def train_lightning(
    model_name,
    datasets,
    num_classes: int = 3,
    num_epochs: int = 1,
    batch_size: int = 32,
):
    CHECKPOINT_PATH = "../tomolint"

    logger = CSVLogger(
        "logs", name=f"run_{model_name}_experiment", flush_logs_every_n_steps=10
    )
    device = (
        torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    )
    num_devices = torch.cuda.device_count()

    datasets.setup("fit")
    datasets.setup("test")

    dataloaders = {
        "train": datasets.train_dataloader(),
        "val": datasets.val_dataloader(),
        "test": datasets.test_dataloader(),
    }

    trainer = lightning.Trainer(
        # default_root_dir= pathlib.Path.cwd() / f"{model_name}",
        default_root_dir=os.path.join(CHECKPOINT_PATH, f"{model_name}"),
        accelerator="gpu" if str(device).startswith("cuda") else "cpu",
        max_epochs=num_epochs,
        log_every_n_steps=1,
        devices=num_devices,
        callbacks=[
            ModelCheckpoint(save_weights_only=True, mode="max", monitor="val_acc"),
            LearningRateMonitor("epoch"),
            EarlyStopping(monitor="val_loss", mode="min"),
        ],
        # resume_from_checkpoint=os.path.join(CHECKPOINT_PATH, f"{model_name}.ckpt"),
        enable_progress_bar=True,
        # logger=WandbLogger(project="tomolint", log_model="all"),
        logger=logger,
    )

    hparams = {
        "vit_params": {
            "embed_dim": 256,
            "hidden_dim": 512,
            "num_heads": 8,
            "num_layers": 6,
            "patch_size": 4,
            "num_channels": 1,
            "num_patches": 64,
            "num_classes": 3,
            "dropout": 0.2,
        },
        "optimizer_params": {
            "lr": 3e-4,
        },
    }

    model = RingClassifier(num_classes, model_name, hparams)

    trainer.fit(
        model,
        dataloaders["train"],
        dataloaders["val"],
    )

    # load checkpoint
    model = RingClassifier.load_from_checkpoint(
        trainer.checkpoint_callback.best_model_path
    )

    return model, [], []


def train(
    model_name,
    datasets,
    num_classes: int = 3,
    num_epochs: int = 1,
    batch_size: int = 32,
) -> torch.nn.Module:
    import logging

    logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)

    datasets.setup("fit")
    datasets.setup("test")

    dataloaders = {
        "train": datasets.train_dataloader(),
        "val": datasets.val_dataloader(),
        "test": datasets.test_dataloader(),
    }

    # Use a pre-trained model

    # model = torchvision.models.resnet18(torchvision.models.ResNet18_Weights.DEFAULT)

    hparams = {
        "vit_params": {
            "embed_dim": 256,
            "hidden_dim": 512,
            "num_heads": 8,
            "num_layers": 6,
            "patch_size": 4,
            "num_channels": 1,
            "num_patches": 64,
            "num_classes": 3,
            "dropout": 0.8,
        },
        "optimizer_params": {
            "lr": 3e-4,
        },
    }
    model = create_model(model_name, hparams)

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
                    # print(f"outputs shape: {outputs.shape}")
                    # print(f"labels shape: {labels.shape}")
                    # outputs should be (batches, classes)
                    # labels should be  (batches, )
                    loss = criterion(outputs, labels)

                    if phase == "train":
                        loss.backward()
                        torch.optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                print(f"running_loss: {running_loss} phase: {phase}")
                running_corrects += torch.sum(preds == labels)

            epoch_loss = running_loss / len(dataloaders[phase])
            epoch_acc = running_corrects.double() / len(dataloaders[phase])
            print(f"Epoch {epoch} {phase} Loss: {epoch_loss} Acc: {epoch_acc}")

            losses[phase].append(epoch_loss)
            accuracies[phase].append(epoch_acc.item())

    return model, losses, accuracies


def train_model(
    model_name,
    datasets,
    num_classes: int = 3,
    num_epochs: int = 1,
    batch_size: int = 32,
    save_name=None,
):
    CHECKPOINT_PATH = "../tomolint"

    datasets.setup("fit")
    datasets.setup("test")

    dataloaders = {
        "train": datasets.train_dataloader(),
        "val": datasets.val_dataloader(),
        "test": datasets.test_dataloader(),
    }

    logger = CSVLogger(
        "logs", name=f"run_{model_name}_experiment", flush_logs_every_n_steps=10
    )
    device = (
        torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    )

    if save_name is None:
        save_name = model_name

    # Create a PyTorch Lightning trainer with the generation callback
    trainer = lightning.Trainer(
        default_root_dir=os.path.join(CHECKPOINT_PATH, save_name),
        accelerator="gpu" if str(device).startswith("cuda") else "cpu",
        devices=1,
        max_epochs=num_epochs,
        callbacks=[
            ModelCheckpoint(save_weights_only=True, mode="max", monitor="val_acc"),
            LearningRateMonitor("epoch"),
        ],
        enable_progress_bar=True,
        logger=logger,
    )
    trainer.logger._log_graph = (
        True  # If True, we plot the computation graph in tensorboard
    )
    trainer.logger._default_hp_metric = (
        None  # Optional logging argument that we don't need
    )

    hparams = {
        "vit_params": {
            "embed_dim": 256,
            "hidden_dim": 512,
            "num_heads": 8,
            "num_layers": 6,
            "patch_size": 4,
            "num_channels": 1,
            "num_patches": 64,
            "num_classes": 3,
            "dropout": 0.2,
        },
        "optimizer_params": {
            "lr": 3e-4,
        },
    }

    # Check whether pretrained model exists. If yes, load it and skip training
    pretrained_filename = os.path.join(CHECKPOINT_PATH, save_name + ".ckpt")
    if os.path.isfile(pretrained_filename):
        print(f"Found pretrained model at {pretrained_filename}, loading...")
        model = RingClassifier(num_classes, model_name, hparams)
        # model = CIFARModule.load_from_checkpoint(
        #     pretrained_filename
        # )  # Automatically loads the model with the saved hyperparameters
    else:
        lightning.seed_everything(42)  # To be reproducable
        model = RingClassifier(num_classes, model_name, hparams)
        trainer.fit(model, dataloaders["train"], dataloaders["val"])

        model = RingClassifier.load_from_checkpoint(
            trainer.checkpoint_callback.best_model_path
        )  # Load best checkpoint after training

    # Test best model on validation and test set
    val_result = trainer.test(model, dataloaders["val"], verbose=False)
    test_result = trainer.test(model, dataloaders["test"], verbose=False)

    accuracies = {"test": test_result[0]["test_acc"], "val": val_result[0]["test_acc"]}

    print(f"Validation accuracy: {result['val']}")
    print(f"Test accuracy: {result['test']}")
    losses = {"train": [], "val": []}
    return model, losses, accuracies

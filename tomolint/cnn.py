import dataclasses
import torch
import torch.nn as nn
import lightning as L


class CNNModel(nn.Module):
    def __init__(self, dim_input=128, dim_hidden=256, num_classes=3, **args):
        super(CNNModel, self).__init__(**args)

        self.layer1 = nn.Conv2d(
            in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1
        )
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.bn1 = nn.BatchNorm2d(32)

        self.layer2 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1
        )
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.bn2 = nn.BatchNorm2d(64)

        self.layer3 = nn.Conv2d(
            in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1
        )
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.bn3 = nn.BatchNorm2d(128)

        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)

        self.fc1 = nn.Linear(128, 256)
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(256, 3)

    def forward(self, x):
        x = self.layer1(x)
        x = self.pool1(x)
        x = self.bn1(x)
        x = self.layer2(x)
        x = self.pool2(x)
        x = self.bn2(x)
        x = self.layer3(x)
        x = self.pool3(x)
        x = self.bn3(x)
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

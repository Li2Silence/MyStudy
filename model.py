# -*- coding: utf-8 -*-
import torch.nn as nn
import torch

class MyModel(nn.Module):
    def __init__(self, num_classes=36, init_weights=False):
        super(MyModel, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 48, kernel_size=(11,11), stride=(4,4), padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(48, 128, kernel_size=(5,5), padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(128, 192, kernel_size=(3,3), padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, kernel_size=(3,3), padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 128, kernel_size=(3,3), padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            # nn.BatchNorm1d(128 * 3 * 3),
            nn.Linear(128 * 3 * 3, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            # nn.BatchNorm1d(512),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)
        return x


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

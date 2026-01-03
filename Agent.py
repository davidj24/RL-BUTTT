import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np


class Agent(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        # backbone convolutional layers
        self.conv1 = nn.Conv2d(7, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        
        # Actor and critic heads
        self.actorfc1 = nn.Linear(in_features=(64*9*9), out_features=512)
        self.actorfc2 = nn.Linear(in_features=(512), out_features=81)

        self.criticfc1 = nn.Linear(in_features=(64*9*9), out_features=512)
        self.criticfc2 = nn.Linear(in_features=(512), out_features=1)


    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv1(x))

        flattened_x = torch.flatten(x, start_dim=1)

        # Policy
        p = F.relu(self.actorfc1(flattened_x))
        logits = self.actorfc2(p)

        # Value
        v = F.relu(self.criticfc1(flattened_x))
        value = self.criticfc2(v)

        return logits, value

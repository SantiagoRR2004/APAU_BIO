#!/usr/bin/env python
import torch.nn as nn
import DQNAgent
import os


class DQN(nn.Module):
    def __init__(self, input_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 128)
        self.fc4 = nn.Linear(128, action_size)

        self.relu = nn.LeakyReLU(0.01)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        return self.fc4(x)


# Usage
train_mode = False  # Change to True if you want to retrain

currentDirectory = os.path.dirname(os.path.abspath(__file__))
# Create the models directory if it doesn't exist
os.makedirs(os.path.join(currentDirectory, "models"), exist_ok=True)

DQNAgent.trainModel(
    "LunarLander-v3",
    dqnClass=DQN,
    fileName=os.path.join(currentDirectory, "models", "dqn_lunar.pth"),
    retrain=train_mode,
)

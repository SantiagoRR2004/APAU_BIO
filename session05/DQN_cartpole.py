#!/usr/bin/env python
import torch.nn as nn
import DQNAgent
import os


class DQN(nn.Module):
    def __init__(self, input_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, 24)
        self.fc2 = nn.Linear(24, 48)
        self.fc3 = nn.Linear(48, action_size)
        self.relu = nn.Tanh()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)  # Linear output (Q-values for each action)


# Usage
train_mode = False

currentDirectory = os.path.dirname(os.path.abspath(__file__))
# Create the models directory if it doesn't exist
os.makedirs(os.path.join(currentDirectory, "models"), exist_ok=True)


DQNAgent.trainModel(
    "CartPole-v1",
    dqnClass=DQN,
    fileName=os.path.join(currentDirectory, "models", "dqn_cartpole.pth"),
    retrain=train_mode,
)

#!/usr/bin/env python
import torch.nn as nn
import DQNAgent


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

if train_mode:
    agent = DQNAgent.Agent(
        "CartPole-v1", render_mode=None, dqnClass=DQN, fileName="dqn_cartpole.pth"
    )
    scores = agent.train_model()
else:
    agent = DQNAgent.Agent(
        "CartPole-v1", render_mode="human", dqnClass=DQN, fileName="dqn_cartpole.pth"
    )
    agent.load_weights_and_visualize()

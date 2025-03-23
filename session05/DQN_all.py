import gymnasium as gym
import DQNAgent
import torch.nn as nn
from collections import defaultdict


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


# Retrieve all environment IDs
env_ids = list(gym.envs.registry.keys())

# Organize environments by base name (without version numbers)
env_dict = defaultdict(list)

for env_id in env_ids:
    base_name, version = env_id.rsplit("-", 1)
    env_dict[base_name].append((int(version[1:]), env_id))  # Convert vX to integer

# Keep only the latest version of each environment
latest_envs = [max(versions, key=lambda x: x[0])[1] for versions in env_dict.values()]
env_ids = sorted(latest_envs)

# Delete GymV21Environment-v0 and GymV26Environment-v0: from the list
env_ids.remove("GymV21Environment-v0")
env_ids.remove("GymV26Environment-v0")


# Print the list of environment IDs
for env_id in env_ids:
    # Get environment specification
    env_spec = gym.spec(env_id)

    # Extract reward threshold (if available)
    threshold = (
        env_spec.reward_threshold
        if env_spec.reward_threshold is not None
        else float("inf")
    )

    print(f"Threshold for {env_id}: {threshold}")

    DQNAgent.trainModel(
        env_id,
        dqnClass=DQN,
        fileName=env_id.replace("/", "_"),
        retrain=False,
        epochs=1000,
    )

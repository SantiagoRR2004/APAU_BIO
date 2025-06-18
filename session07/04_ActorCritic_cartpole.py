import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
import os


# Actor (Policy) Network
class ActorNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(ActorNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return torch.softmax(self.fc2(x), dim=-1)


# Critic (Value) Network
class CriticNetwork(nn.Module):
    def __init__(self, state_size):
        super(CriticNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 1)  # Single output for state value

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


# Actor-Critic Algorithm
def actor_critic(
    env,
    actor_net,
    critic_net,
    actor_optimizer,
    critic_optimizer,
    gamma=0.99,
    num_episodes=1000,
    save_path=None,
):
    actor_losses = []  # To store actor losses
    critic_losses = []  # To store critic losses
    avg_returns = []  # To track average return per episode

    for episode in range(num_episodes):
        state, _ = env.reset()
        done = False
        trajectory = []
        total_reward = 0

        while not done:
            # Convert state to tensor
            state_tensor = torch.tensor(state, dtype=torch.float32)

            # Actor chooses action
            action_probs = actor_net(state_tensor)
            action_dist = torch.distributions.Categorical(action_probs)
            action = action_dist.sample()
            log_prob = action_dist.log_prob(action)  # Log-probability of action

            # Take action in environment
            next_state, reward, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated
            total_reward += reward

            # Convert next state to tensor
            next_state_tensor = torch.tensor(next_state, dtype=torch.float32)

            # Critic computes values
            value_s = critic_net(state_tensor)
            value_s_prime = critic_net(
                next_state_tensor
            ).detach()  # Detach to avoid gradient flow

            # Compute advantage for actor
            td_error = (
                reward + gamma * value_s_prime - value_s
            )  # Temporal Difference (TD) error
            actor_loss = -log_prob * td_error.item()  # Policy gradient loss

            # Update actor network
            actor_optimizer.zero_grad()
            actor_loss.backward()
            actor_optimizer.step()

            # Compute critic loss (TD error squared)
            critic_loss = td_error**2

            # Update critic network
            critic_optimizer.zero_grad()
            critic_loss.backward()
            critic_optimizer.step()

            # Save losses
            actor_losses.append(actor_loss.item())
            critic_losses.append(critic_loss.item())

            # Move to the next state
            state = next_state

        # Log and track performance
        avg_returns.append(total_reward)
        if (episode + 1) % 100 == 0:
            print(
                f"Episode {episode + 1}/{num_episodes}, Average Return: {np.mean(avg_returns[-100:]):.2f}"
            )

    # Save trained parameters
    if save_path:
        torch.save(
            {"actor": actor_net.state_dict(), "critic": critic_net.state_dict()},
            save_path,
        )
        print(f"Model saved to {save_path}")

    # Plot results
    plt.figure(figsize=(10, 5))
    plt.plot(actor_losses, label="Actor Loss")
    plt.plot(critic_losses, label="Critic Loss", color="green")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("Losses Over Time")
    plt.legend()
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(avg_returns, label="Average Return", color="orange")
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.title("Average Return Over Episodes")
    plt.legend()
    plt.show()


# Function to run the environment in prediction mode
def run_human_mode(env, actor_net):
    state, _ = env.reset()
    env.render()
    done = False

    while not done:
        state_tensor = torch.tensor(state, dtype=torch.float32)
        action_probs = actor_net(state_tensor)
        action = torch.argmax(
            action_probs
        ).item()  # Choose action with highest probability

        state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        env.render()


# Main Program
if __name__ == "__main__":
    currentDirectory = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(os.path.join(currentDirectory, "models"), exist_ok=True)
    # Settings
    train = True  # Set to False to load the model and run in human mode
    env_name = "CartPole-v1"  # Change to "FrozenLake-v1" for Frozen Lake
    save_path = os.path.join(currentDirectory, "models", f"{env_name}_actor_critic.pth")

    # Initialize environment
    env = gym.make(env_name, render_mode="human" if not train else None)
    state_size = (
        env.observation_space.shape[0]
        if env_name == "CartPole-v1"
        else env.observation_space.n
    )
    action_size = env.action_space.n

    # Initialize networks
    actor_net = ActorNetwork(state_size, action_size)
    critic_net = CriticNetwork(state_size)

    # Optimizers
    actor_optimizer = optim.Adam(actor_net.parameters(), lr=0.01)
    critic_optimizer = optim.Adam(critic_net.parameters(), lr=0.01)

    if train:
        # Train the networks
        actor_critic(
            env,
            actor_net,
            critic_net,
            actor_optimizer,
            critic_optimizer,
            gamma=0.99,
            num_episodes=1000,
            save_path=save_path,
        )
    else:
        # Load trained models
        if os.path.exists(save_path):
            checkpoint = torch.load(save_path)
            actor_net.load_state_dict(checkpoint["actor"])
            critic_net.load_state_dict(checkpoint["critic"])
            print(f"Model loaded from {save_path}")
        else:
            print(f"No saved model found at {save_path}")
            exit()

        # Run in prediction mode
        run_human_mode(env, actor_net)

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gymnasium as gym
import multiprocessing as mp
import matplotlib.pyplot as plt
import os

# Actor Network
class ActorNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(ActorNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, action_size)  # Actor output
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return torch.softmax(self.fc2(x), dim=-1)

# Critic Network
class CriticNetwork(nn.Module):
    def __init__(self, state_size):
        super(CriticNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 1)  # Critic output
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

# Worker Process
def worker_process(actor_net, critic_net, actor_optimizer, critic_optimizer, env_name, gamma, worker_id, update_freq, actor_losses, critic_losses):
    env = gym.make(env_name)
    local_actor = ActorNetwork(env.observation_space.shape[0], env.action_space.n)
    local_critic = CriticNetwork(env.observation_space.shape[0])

    # Sync local networks with global networks
    local_actor.load_state_dict(actor_net.state_dict())
    local_critic.load_state_dict(critic_net.state_dict())

    state, _ = env.reset()
    done = False
    states, actions, rewards, log_probs = [], [], [], []

    while True:
        # Step through the environment
        state_tensor = torch.tensor(state, dtype=torch.float32)
        policy = local_actor(state_tensor)
        action_dist = torch.distributions.Categorical(policy)
        action = action_dist.sample()
        log_prob = action_dist.log_prob(action)
        
        next_state, reward, terminated, truncated, _ = env.step(action.item())
        done = terminated or truncated
        states.append(state)
        actions.append(action)
        rewards.append(reward)
        log_probs.append(log_prob)

        state = next_state
        if done or len(states) >= update_freq:
            # Compute returns
            if not done:
                with torch.no_grad():
                    next_value = local_critic(torch.tensor(next_state, dtype=torch.float32))
                    rewards[-1] += gamma * next_value.item()

            returns = []
            G = 0
            for reward in reversed(rewards):
                G = reward + gamma * G
                returns.insert(0, G)
            returns = torch.tensor(returns, dtype=torch.float32)

            # Calculate advantages
            states_tensor = torch.tensor(states, dtype=torch.float32)
            values = local_critic(states_tensor).squeeze(1)
            advantages = returns - values

            # Actor loss
            actor_loss = -(torch.stack(log_probs) * advantages.detach()).mean()

            # Critic loss
            critic_loss = advantages.pow(2).mean()

            # Track losses
            actor_losses.append(actor_loss.item())
            critic_losses.append(critic_loss.item())

            # Update global networks
            actor_optimizer.zero_grad()
            critic_optimizer.zero_grad()
            actor_loss.backward()
            critic_loss.backward()

            for global_param, local_param in zip(actor_net.parameters(), local_actor.parameters()):
                global_param._grad = local_param.grad
            for global_param, local_param in zip(critic_net.parameters(), local_critic.parameters()):
                global_param._grad = local_param.grad

            actor_optimizer.step()
            critic_optimizer.step()

            # Sync local networks with global networks
            local_actor.load_state_dict(actor_net.state_dict())
            local_critic.load_state_dict(critic_net.state_dict())

            # Reset buffers
            states, actions, rewards, log_probs = [], [], [], []

            if done:
                state, _ = env.reset()
                done = False

# Main A3C Algorithm
def a3c(env_name, num_workers, gamma=0.99, num_episodes=5000, update_freq=5, lr=0.001, save_path=None):
    env = gym.make(env_name)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    # Global networks
    actor_net = ActorNetwork(state_size, action_size)
    critic_net = CriticNetwork(state_size)
    actor_net.share_memory()
    critic_net.share_memory()

    # Optimizers
    actor_optimizer = optim.Adam(actor_net.parameters(), lr=lr)
    critic_optimizer = optim.Adam(critic_net.parameters(), lr=lr)

    # Shared memory for losses
    actor_losses = mp.Manager().list()
    critic_losses = mp.Manager().list()

    # Spawn worker processes
    processes = []
    for worker_id in range(num_workers):
        process = mp.Process(target=worker_process, args=(
            actor_net, critic_net, actor_optimizer, critic_optimizer, env_name, gamma, worker_id, update_freq, actor_losses, critic_losses))
        processes.append(process)
        process.start()

    # Wait for all processes to complete
    for process in processes:
        process.join()

    # Save trained parameters
    if save_path:
        torch.save({
            "actor": actor_net.state_dict(),
            "critic": critic_net.state_dict()
        }, save_path)
        print(f"Model saved to {save_path}")

    # Plot results
    plt.figure(figsize=(10, 5))
    plt.plot(actor_losses, label="Actor Loss")
    plt.plot(critic_losses, label="Critic Loss", color="green")
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.title("Actor and Critic Losses Over Time")
    plt.legend()
    plt.show()

# Run in Human Mode
def run_human_mode(env_name, save_path):
    env = gym.make(env_name, render_mode="human")
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    # Load trained networks
    actor_net = ActorNetwork(state_size, action_size)
    checkpoint = torch.load(save_path)
    actor_net.load_state_dict(checkpoint["actor"])
    print(f"Model loaded from {save_path}")

    state, _ = env.reset()
    done = False
    while not done:
        state_tensor = torch.tensor(state, dtype=torch.float32)
        policy = actor_net(state_tensor)
        action = torch.argmax(policy).item()
        state, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        env.render()

# Main Entry Point
if __name__ == "__main__":
    train = True  # Set to False for human mode
    env_name = "CartPole-v1"
    save_path = f"{env_name}_a3c.pth"

    if train:
        num_workers = mp.cpu_count() - 1  # Use all available CPUs minus one
        a3c(env_name, num_workers, save_path=save_path)
    else:
        if os.path.exists(save_path):
            run_human_mode(env_name, save_path)
        else:
            print(f"No saved model found at {save_path}")

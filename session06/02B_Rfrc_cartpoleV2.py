import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
import os

# -----------------------------------------------------------------------------
# 1. Policy Network for CartPole
# -----------------------------------------------------------------------------
class PolicyNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)  # Hidden layer size is adjustable
        self.fc2 = nn.Linear(128, action_size)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return torch.softmax(self.fc2(x), dim=-1)

# -----------------------------------------------------------------------------
# 2. Generate a single trajectory by sampling from the current policy
# -----------------------------------------------------------------------------
def generate_trajectory(env, policy_net):
    state, _ = env.reset()
    trajectory = []
    done = False
    
    while not done:
        # Convert state to tensor
        state_tensor = torch.tensor(state, dtype=torch.float32)
        # Get action probabilities
        action_probs = policy_net(state_tensor)
        action_dist = torch.distributions.Categorical(action_probs)
        action = action_dist.sample()           # Sample an action
        log_prob = action_dist.log_prob(action) # Log-probability of chosen action
        
        # Step in the environment
        next_state, reward, terminated, truncated, _ = env.step(action.item())
        done = terminated or truncated
        
        # Store transition
        trajectory.append((state, action.item(), log_prob, reward))
        state = next_state
    
    return trajectory

# -----------------------------------------------------------------------------
# 3. Compute returns (G) for a given trajectory
# -----------------------------------------------------------------------------
def compute_returns(trajectory, gamma):
    """ G_t = r_t + gamma * r_{t+1} + ... """
    returns = []
    G = 0
    for _, _, _, reward in reversed(trajectory):
        G = reward + gamma * G
        returns.insert(0, G)  # Insert at the front
    return returns

# -----------------------------------------------------------------------------
# 4. REINFORCE Algorithm with Return Normalization and live plotting
# -----------------------------------------------------------------------------
def reinforce(env, policy_net, optimizer, gamma=0.99, num_iterations=1000, N=10, save_path=None):
    """
    :param env: Gym environment
    :param policy_net: PolicyNetwork (nn.Module)
    :param optimizer: Torch optimizer
    :param gamma: Discount factor
    :param num_iterations: Number of training iterations
    :param N: Number of trajectories per iteration
    :param save_path: If provided, file path to save the policy_net at end
    """
    losses = []      # Stores policy loss each iteration
    avg_returns = [] # Stores average returns across N trajectories each iteration
    
    # Enable interactive plotting
    plt.ion()
    fig, ax = plt.subplots(2, 1, figsize=(10, 10))

    # Subplot 1: Policy Loss
    loss_plot, = ax[0].plot([], [], label="Policy Loss")
    ax[0].set_title("Policy Loss Over Iterations")
    ax[0].set_xlabel("Iteration")
    ax[0].set_ylabel("Loss")
    ax[0].legend()

    # Subplot 2: Average Return
    return_plot, = ax[1].plot([], [], label="Average Return", color="orange")
    ax[1].set_title("Average Return Over Iterations")
    ax[1].set_xlabel("Iteration")
    ax[1].set_ylabel("Return")
    ax[1].legend()

    for iteration in range(num_iterations):
        trajectories = []
        all_returns = []
        
        # Generate N trajectories
        for _ in range(N):
            trajectory = generate_trajectory(env, policy_net)
            trajectories.append(trajectory)
            
            # Compute returns for each trajectory
            returns = compute_returns(trajectory, gamma)
            all_returns.append(returns)
        
        # Flatten all returns (across all trajectories) to compute mean/std
        all_returns_flat = [G for returns in all_returns for G in returns]
        mean_return = np.mean(all_returns_flat)
        std_return = np.std(all_returns_flat) + 1e-8  # Avoid div by zero
       
        # Normalize returns
        normalized_returns_list = [
            [(G - mean_return) / std_return for G in returns]
            for returns in all_returns
        ]

        # Compute policy gradients
        policy_loss = 0.0
        for trajectory, normalized_returns in zip(trajectories, normalized_returns_list):
            for (state, action, log_prob, _), G in zip(trajectory, normalized_returns):
                policy_loss += -log_prob * G  # REINFORCE objective
        
        policy_loss /= N  # Average over N trajectories

        # Update network parameters
        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()

        # Record the policy loss
        losses.append(policy_loss.item())

        # Compute average return from these N trajectories
        avg_return = np.mean([sum([r for (_, _, _, r) in traj]) for traj in trajectories])
        avg_returns.append(avg_return)

        # Print log every 100 iterations (optional)
        if (iteration + 1) % 100 == 0:
            print(f"Iteration {iteration + 1}/{num_iterations} | Avg Return: {avg_return:.2f}")

        # Update plots
        loss_plot.set_data(range(len(losses)), losses)
        return_plot.set_data(range(len(avg_returns)), avg_returns)

        # Rescale the axes
        for i in range(2):
            ax[i].relim()
            ax[i].autoscale_view()

        plt.pause(0.01)  # Pause briefly to update the figure

    # Turn off interactive mode and show final plot
    plt.ioff()
    plt.show()

    # Optionally save the model
    if save_path is not None:
        torch.save(policy_net.state_dict(), save_path)
        print(f"Model saved to {save_path}")

# -----------------------------------------------------------------------------
# 5. Run environment in human mode with a greedy policy
# -----------------------------------------------------------------------------
def run_human_mode(env, policy_net):
    state, _ = env.reset()
    env.render()
    done = False

    while not done:
        state_tensor = torch.tensor(state, dtype=torch.float32)
        action_probs = policy_net(state_tensor)
        action = torch.argmax(action_probs).item()  # Greedy w.r.t. highest probability
        
        state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        env.render()

# -----------------------------------------------------------------------------
# Main Script
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    train = True  # Set to False to load a model and run in human mode
    env_name = "CartPole-v1"
    save_path = f"{env_name}_policy.pth"

    # Create environment
    env = gym.make(env_name, render_mode="human" if not train else None)

    # Determine state and action sizes
    if env_name == "CartPole-v1":
        state_size = env.observation_space.shape[0]
        action_size = env.action_space.n
    else:
        # Example: if using discrete states (like FrozenLake), adapt accordingly
        state_size = env.observation_space.n
        action_size = env.action_space.n

    # Initialize policy
    policy_net = PolicyNetwork(state_size, action_size)

    if train:
        # Train the policy
        optimizer = optim.Adam(policy_net.parameters(), lr=0.01)
        reinforce(env, policy_net, optimizer,
                  gamma=0.99,
                  num_iterations=100,   # Increase if needed
                  N=10,
                  save_path=save_path)
    else:
        # Load a previously saved policy
        if os.path.exists(save_path):
            policy_net.load_state_dict(torch.load(save_path))
            print(f"Model loaded from {save_path}")
        else:
            print(f"No saved model found at {save_path}, please train first.")
            exit()

        # Run environment in human mode
        run_human_mode(env, policy_net)

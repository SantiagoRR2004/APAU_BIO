import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
import os

# -----------------------------------------------------------------------------
# 1. Policy Network for FrozenLake
# -----------------------------------------------------------------------------
class PolicyNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)  # Hidden layer size
        self.fc2 = nn.Linear(64, action_size)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return torch.softmax(self.fc2(x), dim=-1)

# -----------------------------------------------------------------------------
# 2. Generate a single trajectory by sampling from the current policy
# -----------------------------------------------------------------------------
def generate_trajectory(env, policy_net):
    """
    Returns a list of transitions: (state, action, log_prob, reward)
    for one episode in the environment.
    """
    state, _ = env.reset()
    trajectory = []
    done = False
    
    while not done:
        # For FrozenLake with discrete states, use one-hot encoding
        state_tensor = torch.eye(env.observation_space.n)[state]  # shape [n_states]
        
        # Get action probabilities
        action_probs = policy_net(state_tensor)
        action_dist = torch.distributions.Categorical(action_probs)
        action = action_dist.sample()           # Sample an action
        log_prob = action_dist.log_prob(action) # Log-probability of chosen action
        
        # Step in the environment
        next_state, reward, terminated, truncated, _ = env.step(action.item())
        done = terminated or truncated

        # Store the transition
        trajectory.append((state, action.item(), log_prob, reward))
        
        # Move to next state
        state = next_state
    
    return trajectory

# -----------------------------------------------------------------------------
# 3. Compute returns for a trajectory
# -----------------------------------------------------------------------------
def compute_returns(trajectory, gamma):
    """Return a list of discounted returns G_t for each step in the trajectory."""
    returns = []
    G = 0.0
    for _, _, _, reward in reversed(trajectory):
        G = reward + gamma * G
        returns.insert(0, G)  # Insert at front
    return returns

# -----------------------------------------------------------------------------
# 4. REINFORCE Algorithm with Interactive Plotting + Optional Model Save
# -----------------------------------------------------------------------------
def reinforce(
    env,
    policy_net,
    optimizer,
    gamma=0.99,
    num_iterations=1000,
    N=10,
    save_path=None
):
    """
    Train the policy using REINFORCE with multiple (N) trajectories per iteration.
    
    :param env: Gym environment
    :param policy_net: Torch nn.Module (PolicyNetwork)
    :param optimizer: Torch optimizer (e.g., Adam)
    :param gamma: Discount factor
    :param num_iterations: Total training iterations
    :param N: Number of trajectories to collect each iteration
    :param save_path: (optional) file path to save the policy after training
    """
    losses = []       # To store the loss at each iteration
    avg_returns = []  # To track average return at each iteration
    
    # Enable interactive plotting
    plt.ion()
    fig, ax = plt.subplots(2, 1, figsize=(10, 10))

    # Subplot 1: Policy Loss
    (loss_plot,) = ax[0].plot([], [], label="Policy Loss")
    ax[0].set_title("Policy Loss Over Iterations")
    ax[0].set_xlabel("Iteration")
    ax[0].set_ylabel("Loss")
    ax[0].legend()

    # Subplot 2: Average Return
    (return_plot,) = ax[1].plot([], [], label="Average Return", color="orange")
    ax[1].set_title("Average Return Over Iterations")
    ax[1].set_xlabel("Iteration")
    ax[1].set_ylabel("Average Return")
    ax[1].legend()

    for iteration in range(num_iterations):
        trajectories = []
        all_returns = []
        
        # Generate N trajectories
        for _ in range(N):
            trajectory = generate_trajectory(env, policy_net)
            trajectories.append(trajectory)
            
            # Compute returns
            returns = compute_returns(trajectory, gamma)
            all_returns.append(returns)
        
        # Compute the policy loss
        policy_loss = 0.0
        for trajectory, returns in zip(trajectories, all_returns):
            for (_, _, log_prob, _), G in zip(trajectory, returns):
                policy_loss += -log_prob * G  # REINFORCE update
                
        policy_loss /= N  # Average over N trajectories

        # Gradient ascent step
        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()

        # Track loss
        losses.append(policy_loss.item())

        # Compute average return for these N trajectories
        avg_return = np.mean([sum([r for (_, _, _, r) in traj]) for traj in trajectories])
        avg_returns.append(avg_return)
        
        # Logging every 50 iterations
        if (iteration + 1) % 50 == 0:
            print(f"Iteration {iteration+1}/{num_iterations}, Average Return: {avg_return:.2f}")

        # Update plots
        loss_plot.set_data(range(len(losses)), losses)
        return_plot.set_data(range(len(avg_returns)), avg_returns)

        # Auto-rescale
        ax[0].relim()
        ax[0].autoscale_view()
        ax[1].relim()
        ax[1].autoscale_view()
        plt.pause(0.01)  # Short pause to update the figure

    plt.ioff()
    plt.show()

    # Optionally save the trained policy network
    if save_path is not None:
        torch.save(policy_net.state_dict(), save_path)
        print(f"Model saved to {save_path}")

# -----------------------------------------------------------------------------
# 5. Run the environment in "human mode" to visualize the learned policy
# -----------------------------------------------------------------------------
def run_human_mode(env, policy_net):
    """
    Let the user watch a few episodes in the console (text-based) with the
    best action chosen at each step (argmax of the learned policy).
    """
    # For demonstration, run 5 episodes
    for episode in range(5):
        state, _ = env.reset()
        done = False
        print(f"\n=== Starting episode {episode+1} ===\n")
        
        while not done:
            env.render()  # text-based output for FrozenLake
            # One-hot encode the discrete state
            state_tensor = torch.eye(env.observation_space.n)[state]
            # Choose action greedily w.r.t. policy
            action_probs = policy_net(state_tensor)
            action = torch.argmax(action_probs).item()
            
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            state = next_state
            
            if done:
                env.render()
                print(f"Episode ended with reward = {reward}\n")

# -----------------------------------------------------------------------------
# Main Script
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # Decide whether we are training or loading a saved model
    train = True  # Set to False to load a model and run in human mode
    
    env_name = "FrozenLake-v1"
    # For demonstration, we fix is_slippery=False (deterministic environment)
    env = gym.make(env_name, is_slippery=False, render_mode=None if train else "human")
    
    state_size = env.observation_space.n  # discrete states
    action_size = env.action_space.n      # discrete actions
    policy_net = PolicyNetwork(state_size, action_size)
    
    # Path to save/load the policy model
    save_path = "frozenlake_policy.pth"
    
    if train:
        # We will train the REINFORCE agent
        optimizer = optim.Adam(policy_net.parameters(), lr=0.01)
        reinforce(env,
                  policy_net,
                  optimizer,
                  gamma=0.99,
                  num_iterations=100,  # Increase if you like
                  N=10,
                  save_path=save_path)
    else:
        # Load existing policy
        if os.path.isfile(save_path):
            policy_net.load_state_dict(torch.load(save_path))
            print(f"Loaded policy from {save_path}")
        else:
            raise FileNotFoundError(f"No saved model found at {save_path}. Please train first.")
        
        # Re-create the environment in "human" mode for textual output
        env = gym.make(env_name, is_slippery=False, render_mode="human")
        
        # Run several episodes with the trained policy in human mode
        run_human_mode(env, policy_net)

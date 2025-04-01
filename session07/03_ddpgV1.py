import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import matplotlib.pyplot as plt

# -------------------------------------------------
# 1) Replay Buffer
# -------------------------------------------------
class ReplayBuffer:
    """
    Stores tuples of (state, action, reward, next_state, done)
    for training the DDPG agent.
    """
    def __init__(self, max_size, state_dim, action_dim):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        # Buffers
        self.states = np.zeros((max_size, state_dim), dtype=np.float32)
        self.actions = np.zeros((max_size, action_dim), dtype=np.float32)
        self.rewards = np.zeros((max_size, 1), dtype=np.float32)
        self.next_states = np.zeros((max_size, state_dim), dtype=np.float32)
        self.dones = np.zeros((max_size, 1), dtype=np.float32)

    def store(self, state, action, reward, next_state, done):
        """
        Add a new transition to the replay buffer.
        """
        idx = self.ptr

        self.states[idx] = state
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.next_states[idx] = next_state
        self.dones[idx] = float(done)

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self, batch_size=64):
        """
        Sample a random batch of transitions for training.
        Returns tensors ready for PyTorch.
        """
        idxs = np.random.randint(0, self.size, size=batch_size)

        states = torch.tensor(self.states[idxs], dtype=torch.float32)
        actions = torch.tensor(self.actions[idxs], dtype=torch.float32)
        rewards = torch.tensor(self.rewards[idxs], dtype=torch.float32)
        next_states = torch.tensor(self.next_states[idxs], dtype=torch.float32)
        dones = torch.tensor(self.dones[idxs], dtype=torch.float32)

        return states, actions, rewards, next_states, dones


# -------------------------------------------------
# 2) Actor and Critic Networks
# -------------------------------------------------
class ActorNetwork(nn.Module):
    """
    Maps states -> actions, using a tanh output to keep them in range.
    For Pendulum, the action space is 1D within [-2, 2].
    We'll multiply the tanh output by max_action in the forward() method.
    """
    def __init__(self, state_dim, action_dim, max_action):
        super(ActorNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.out = nn.Linear(256, action_dim)
        self.max_action = max_action

        # Optional: weight initialization, etc.

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # Output in [-1, 1], then scale by max_action
        x = torch.tanh(self.out(x))
        return x * self.max_action


class CriticNetwork(nn.Module):
    """
    Maps (state, action) -> Q-value.
    We'll concatenate state and action at the first layer.
    """
    def __init__(self, state_dim, action_dim):
        super(CriticNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.out = nn.Linear(256, 1)

    def forward(self, state, action):
        # Concatenate along dimension=1
        x = torch.cat([state, action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out(x)
        return x


# -------------------------------------------------
# 3) DDPG Agent Class
# -------------------------------------------------
class DDPGAgent:
    """
    The DDPG agent maintains:
      - Actor & Critic networks (and their target networks)
      - Replay buffer
      - Hyperparameters for training
      - Noise (exploration)
    """
    def __init__(
        self,
        state_dim,
        action_dim,
        max_action,
        lr_actor=1e-3,
        lr_critic=1e-3,
        gamma=0.99,
        tau=0.005,
        buffer_size=1000000,
        batch_size=64
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action

        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size

        # Create main networks
        self.actor = ActorNetwork(state_dim, action_dim, max_action)
        self.critic = CriticNetwork(state_dim, action_dim)

        # Create target networks
        self.target_actor = ActorNetwork(state_dim, action_dim, max_action)
        self.target_critic = CriticNetwork(state_dim, action_dim)

        # Copy params from main networks to target networks
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())

        # Create optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)

        # Replay buffer
        self.replay_buffer = ReplayBuffer(buffer_size, state_dim, action_dim)

        # Exploration noise parameters (Ornstein-Uhlenbeck or simple Gaussian)
        self.noise_std = 0.1  # standard deviation for Gaussian noise

    def select_action(self, state, explore=True):
        """
        Given a state, returns an action (optionally with exploration noise).
        state: np array of shape (state_dim,)
        """
        state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            action = self.actor(state_t).cpu().numpy().flatten()  # shape (action_dim,)

        if explore:
            # Add random Gaussian noise
            noise = np.random.normal(0, self.noise_std, size=self.action_dim)
            action = action + noise

        # clip to max_action (for safety)
        action = np.clip(action, -self.max_action, self.max_action)
        return action

    def store_transition(self, state, action, reward, next_state, done):
        """
        Save a transition to the replay buffer.
        """
        self.replay_buffer.store(state, action, reward, next_state, done)

    def train_one_step(self):
        """
        Sample a batch from the buffer, update the Critic, then update the Actor,
        then do a soft update for target networks.
        """
        if self.replay_buffer.size < self.batch_size:
            # Not enough samples yet
            return

        states, actions, rewards, next_states, dones = self.replay_buffer.sample_batch(
            self.batch_size
        )

        # ---------------------
        # Critic update
        # ---------------------
        with torch.no_grad():
            # target actions from target_actor
            next_actions = self.target_actor(next_states)
            # target Q
            target_Q = self.target_critic(next_states, next_actions)
            # Bellman backup
            target = rewards + self.gamma * (1 - dones) * target_Q

        # current Q estimate
        current_Q = self.critic(states, actions)
        critic_loss = F.mse_loss(current_Q, target)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # ---------------------
        # Actor update
        # ---------------------
        # We want to maximize Q(state, actor(state)),
        # so we do gradient ascent on Q w.r.t. actor params
        actor_actions = self.actor(states)
        actor_loss = -self.critic(states, actor_actions).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ---------------------
        # Soft update target networks
        # ---------------------
        self.soft_update(self.target_actor, self.actor)
        self.soft_update(self.target_critic, self.critic)

    def soft_update(self, target_net, source_net):
        """
        θ_target = τ*θ_source + (1 - τ)*θ_target
        """
        for target_param, source_param in zip(target_net.parameters(), source_net.parameters()):
            target_param.data.copy_(
                self.tau * source_param.data + (1.0 - self.tau) * target_param.data
            )


# -------------------------------------------------
# 4) Training Loop (Gymnasium) + Plot
# -------------------------------------------------
def train_ddpg_on_pendulum(
    env_name="Pendulum-v1",
    num_episodes=200,
    max_steps=200,
    render=False
):
    """
    Train a DDPG agent on the given Gymnasium continuous-control environment.
    For demonstration, uses Pendulum-v1 by default.
    """
    # Create environment
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    # Pendulum has actions in [-2, 2]
    max_action = float(env.action_space.high[0])  # Typically 2.0 for Pendulum

    # Create DDPG Agent
    agent = DDPGAgent(state_dim, action_dim, max_action)

    returns_history = []

    for episode in range(num_episodes):
        state, info = env.reset()
        episode_return = 0.0

        for step in range(max_steps):
            if render:
                env.render()

            # Select action (with exploration)
            action = agent.select_action(state, explore=True)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Store transition in replay buffer
            agent.store_transition(state, action, reward, next_state, done)

            # Update the agent
            agent.train_one_step()

            state = next_state
            episode_return += reward

            if done:
                break

        returns_history.append(episode_return)
        print(f"Episode {episode+1}/{num_episodes}, Return: {episode_return:.2f}")

    # Plot returns
    plt.plot(returns_history)
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.title(f"DDPG on {env_name}")
    plt.show()

    env.close()


# -------------------------------------------------
# 5) Main Entry
# -------------------------------------------------
if __name__ == "__main__":
    train_ddpg_on_pendulum(
        env_name="Pendulum-v1", 
        num_episodes=200, 
        max_steps=200, 
        render=False
    )

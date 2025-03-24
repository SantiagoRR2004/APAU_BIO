#!/usr/bin/env python
import os
import numpy as np
import matplotlib.pyplot as plt
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim

import gymnasium as gym
from gymnasium.envs.box2d import LunarLander

# For DQN approach:
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

###############################################################################
# 1) Custom Environment: Subclassing LunarLander with a Fuel Penalty
###############################################################################
class CustomLunarLanderEnv(LunarLander):
    def __init__(self, fuel_penalty_multiplier=2.0, **kwargs):
        """
        :param fuel_penalty_multiplier: Factor to multiply fuel usage penalty.
        :param kwargs: Additional arguments for the LunarLander parent.
        """
        super().__init__(**kwargs)
        self.fuel_penalty_multiplier = fuel_penalty_multiplier

    def step(self, action):
        # Original step from LunarLander
        state, reward, terminated, truncated, info = super().step(action)
        # Apply custom penalty
        reward = self.custom_reward_function(state, reward, action)
        return state, reward, terminated, truncated, info

    def custom_reward_function(self, state, reward, action):
        # Example: penalize action 2 (main engine) more heavily
        # and actions 1 or 3 (side engines) lightly
        fuel_usage = 0.0
        if action == 1 or action == 3:
            fuel_usage += 0.03
        elif action == 2:
            fuel_usage += 0.3

        reward -= self.fuel_penalty_multiplier * fuel_usage
        return reward

###############################################################################
# 2) DQN Approach (Stable-Baselines3)
###############################################################################
class RewardTrackingCallback(BaseCallback):
    """
    A custom callback to log and plot episode rewards during DQN training.
    """
    def __init__(self, save_path='results/lunarlander_rewards.csv', verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.episodes = []
        self.episode_count = 0
        self.last_100_rewards = deque(maxlen=100)

        # File for saving rewards
        self.filepath = save_path
        if not os.path.exists('results'):
            os.makedirs('results')
        with open(self.filepath, 'w') as f:
            f.write("Episode,Reward,Length\n")

        # Setup plotting
        self.fig, self.ax = plt.subplots()
        self.ax.set_xlabel("Episode")
        self.ax.set_ylabel("Reward")
        plt.ion()  # interactive mode

    def _on_step(self) -> bool:
        # Collect episode info from 'infos'
        infos = self.locals.get('infos', [])
        for info in infos:
            if 'episode' in info:
                ep_reward = info['episode']['r']
                ep_length = info['episode']['l']
                self.episode_rewards.append(ep_reward)
                self.episode_lengths.append(ep_length)
                self.episodes.append(self.episode_count)
                self.last_100_rewards.append(ep_reward)
                self.episode_count += 1

                # Save to CSV
                with open(self.filepath, 'a') as f:
                    f.write(f"{self.episode_count},{ep_reward},{ep_length}\n")

                print(f"Episode {self.episode_count}: Reward={ep_reward:.2f}, Length={ep_length}")

                # Update live plot
                self.ax.clear()
                self.ax.set_xlabel("Episode")
                self.ax.set_ylabel("Reward")
                self.ax.plot(self.episodes, self.episode_rewards, label="Episode Reward")
                self.ax.legend()
                plt.draw()
                plt.pause(0.001)

                # Save figure
                plt.savefig('results/reward_plot.png')

        return True

class DQNTrainer:
    """
    Encapsulates training/evaluation of a DQN on CustomLunarLanderEnv.
    """
    def __init__(self,
                 fuel_penalty_multiplier=2.0,
                 render_mode=None,
                 total_timesteps=100_000):
        self.fuel_penalty_multiplier = fuel_penalty_multiplier
        self.render_mode = render_mode
        self.total_timesteps = total_timesteps
        self.env = None
        self.model = None

    def create_env(self):
        env = CustomLunarLanderEnv(fuel_penalty_multiplier=self.fuel_penalty_multiplier,
                                   render_mode=self.render_mode)
        env = Monitor(env)
        return env

    def train(self):
        self.env = self.create_env()
        from stable_baselines3 import DQN

        self.model = DQN(
            policy='MlpPolicy',
            env=self.env,
            verbose=1,
            tensorboard_log="./logs/"
        )

        callback = RewardTrackingCallback()
        self.model.learn(total_timesteps=self.total_timesteps, callback=callback)
        self.model.save("dqn_custom_lunar_lander")
        print("DQN model saved to dqn_custom_lunar_lander.zip")
        self.env.close()

    def evaluate(self, deterministic=True):
        # Re-create env with human rendering
        self.env = self.create_env()
        self.env.render_mode = 'human'

        from stable_baselines3 import DQN
        if self.model is None:
            self.model = DQN.load("dqn_custom_lunar_lander")

        obs, info = self.env.reset()
        done = False
        while not done:
            action, _ = self.model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            self.env.render()

        self.env.close()

###############################################################################
# 3) Policy Gradient (REINFORCE) Approach
###############################################################################
class PolicyNet(nn.Module):
    """
    Simple MLP policy for discrete action spaces.
    Observations: shape (8,) for LunarLander
    Outputs: action_probs of shape (4,)
    """
    def __init__(self, input_size=8, action_size=4, hidden_size=128):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return torch.softmax(self.fc2(x), dim=-1)

def generate_trajectory(env, policy):
    """
    Returns a list of (state, action, log_prob, reward) for one complete episode.
    """
    state, _ = env.reset()
    trajectory = []
    done = False
    while not done:
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            action_probs = policy(state_tensor).squeeze(0)  # shape (4,)
        dist = torch.distributions.Categorical(action_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        next_state, reward, terminated, truncated, info = env.step(action.item())
        done = terminated or truncated

        trajectory.append((state, action.item(), log_prob, reward))
        state = next_state
    return trajectory

def compute_returns(trajectory, gamma=0.99):
    """
    Compute discounted returns G_t from the final step backward.
    """
    returns = []
    G = 0.0
    for _, _, _, reward in reversed(trajectory):
        G = reward + gamma * G
        returns.insert(0, G)
    return returns

class REINFORCETrainer:
    """
    Manual REINFORCE approach on the same CustomLunarLanderEnv.
    """
    def __init__(self,
                 fuel_penalty_multiplier=2.0,
                 render_mode=None,
                 lr=1e-3,
                 gamma=0.99,
                 num_iterations=5000,
                 batch_size_episodes=5):
        """
        :param batch_size_episodes: Number of episodes per policy update
        """
        self.fuel_penalty_multiplier = fuel_penalty_multiplier
        self.render_mode = render_mode
        self.lr = lr
        self.gamma = gamma
        self.num_iterations = num_iterations
        self.batch_size_episodes = batch_size_episodes

        self.env = None
        self.policy = None
        self.optimizer = None

        # Logging
        if not os.path.exists('results'):
            os.makedirs('results')
        self.rewards_log = []
        self.fig, self.ax = plt.subplots()
        plt.ion()

    def create_env(self):
        env = CustomLunarLanderEnv(
            fuel_penalty_multiplier=self.fuel_penalty_multiplier,
            render_mode=self.render_mode
        )
        return Monitor(env)

    def train(self):
        """
        Train using REINFORCE:
        1) Collect N episodes.
        2) Compute discounted returns, optionally normalize.
        3) Accumulate policy gradient updates.
        4) Update weights.
        """
        self.env = self.create_env()
        obs_space = self.env.observation_space
        act_space = self.env.action_space

        input_size = obs_space.shape[0]  # e.g. 8 for LunarLander
        action_size = act_space.n        # e.g. 4
        self.policy = PolicyNet(input_size, action_size)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.lr)

        iteration_rewards = []
        episodes_collected = 0

        for iteration in range(self.num_iterations):
            batch_trajectories = []
            batch_returns = []

            # 1) Collect a batch of episodes
            for _ in range(self.batch_size_episodes):
                traj = generate_trajectory(self.env, self.policy)
                batch_trajectories.append(traj)
                # Compute discounted returns
                Gs = compute_returns(traj, self.gamma)
                batch_returns.append(Gs)
                episodes_collected += 1

            # 2) Policy gradient update
            loss = 0.0
            for traj, Gs in zip(batch_trajectories, batch_returns):
                for (state, action, log_prob, _), G in zip(traj, Gs):
                    loss += -log_prob * G
            loss /= self.batch_size_episodes

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # 3) Logging
            # Average reward across the batch
            avg_return = np.mean([sum([r for (_,_,_, r) in traj]) for traj in batch_trajectories])
            iteration_rewards.append(avg_return)
            self.rewards_log.append(avg_return)

            if (iteration+1) % 10 == 0:
                print(f"Iteration {iteration+1}/{self.num_iterations}, "
                      f"AvgReturn={avg_return:.2f}, Loss={loss.item():.3f}")

            # 4) Update live plot
            self.ax.clear()
            self.ax.set_title("REINFORCE: Average Return")
            self.ax.set_xlabel("Iteration")
            self.ax.set_ylabel("Return")
            self.ax.plot(range(len(iteration_rewards)), iteration_rewards)
            plt.draw()
            plt.pause(0.001)
            plt.savefig('results/reinforce_reward_plot.png')

        plt.ioff()
        plt.show()
        print("REINFORCE Training finished.")

        # Save final policy
        torch.save(self.policy.state_dict(), "reinforce_lunar_lander.pth")
        print("Saved policy to reinforce_lunar_lander.pth")

        self.env.close()

    def evaluate(self, episodes=5):
        """
        Evaluate the policy in 'human' mode for a certain number of episodes.
        """
        if self.policy is None:
            # Load from disk if needed
            self.policy = PolicyNet()
            self.policy.load_state_dict(torch.load("reinforce_lunar_lander.pth"))
        self.env = self.create_env()
        self.env.render_mode = 'human'

        for ep in range(episodes):
            state, _ = self.env.reset()
            done = False
            total_reward = 0.0
            while not done:
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                with torch.no_grad():
                    action_probs = self.policy(state_tensor).squeeze(0)
                dist = torch.distributions.Categorical(action_probs)
                action = dist.sample()
                state, reward, terminated, truncated, info = self.env.step(action.item())
                done = terminated or truncated
                self.env.render()
                total_reward += reward
            print(f"Episode {ep+1} ended with reward={total_reward:.2f}")
        self.env.close()

###############################################################################
# 4) Main: Select DQN or REINFORCE
###############################################################################
if __name__ == "__main__":
    """
    You can choose which method to use:
      - "dqn" for stable-baselines3 DQN
      - "pg"  for manual REINFORCE
    """
    method = "dqn"  # or "pg"

    if method == "dqn":
        agent = DQNTrainer(fuel_penalty_multiplier=2.0)
        train_mode = True
        if train_mode:
            agent.train()       # Train DQN
        else:
            agent.evaluate()    # Evaluate DQN
    else:
        # Policy Gradient approach
        pg_agent = REINFORCETrainer(
            fuel_penalty_multiplier=2.0,
            lr=1e-3,
            gamma=0.99,
            num_iterations=500,      # Increase for better results
            batch_size_episodes=5
        )
        train_mode = True
        if train_mode:
            pg_agent.train()    # Train REINFORCE
        else:
            pg_agent.evaluate(episodes=5)  # Evaluate REINFORCE

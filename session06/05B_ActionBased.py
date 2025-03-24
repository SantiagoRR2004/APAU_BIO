import os
import numpy as np
import matplotlib.pyplot as plt
from collections import deque

import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor


class ActionBasedRewardWrapper(gym.RewardWrapper):
    """
    Custom reward wrapper that penalizes fuel usage.
    """

    def __init__(self, env, fuel_penalty_multiplier=2.0):
        super(ActionBasedRewardWrapper, self).__init__(env)
        self.fuel_penalty_multiplier = fuel_penalty_multiplier

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        # Modify the reward based on the action (fuel usage)
        reward = self.reward(reward, action)
        return observation, reward, terminated, truncated, info

    def reward(self, reward, action):
        # Example: main engine uses more fuel, so higher penalty
        fuel_usage = 0.0
        if action == 1 or action == 3:
            fuel_usage += 0.03
        elif action == 2:
            fuel_usage += 0.3

        # Apply multiplier
        fuel_penalty = self.fuel_penalty_multiplier * fuel_usage
        reward -= fuel_penalty
        return reward


class LunarLanderAgent:
    """
    Encapsulates training and evaluation of a DQN on LunarLander with a custom
    ActionBasedRewardWrapper and a callback for plotting rewards.
    """

    def __init__(self, train_mode=True):
        self.train_mode = train_mode
        self.env = None
        self.model = None
        self.callback = None

        # Create directory to store results if not exists
        if not os.path.exists("results"):
            os.makedirs("results")

    def create_env(self, render_mode=None):
        # Use the updated 'LunarLander-v3' environment
        env = gym.make("LunarLander-v3", render_mode=render_mode)
        # Wrap the environment with our custom reward wrapper
        env = ActionBasedRewardWrapper(env)
        # Wrap the environment with a Monitor to track episode stats
        env = Monitor(env)
        return env

    def create_model(self):
        # Create the DQN model (MlpPolicy) with optional TensorBoard logging
        self.model = DQN(
            policy="MlpPolicy", env=self.env, verbose=1, tensorboard_log="./logs/"
        )

    def train(self, total_timesteps=100000):
        # Create the environment
        self.env = self.create_env()
        # Create the model
        self.create_model()
        # Create the callback that plots and saves rewards
        self.callback = self.RewardTrackingCallback()

        # Train the model
        self.model.learn(total_timesteps=total_timesteps, callback=self.callback)
        # Save the trained model
        self.model.save("dqn_custom_lunar_lander")
        print("Model saved to dqn_custom_lunar_lander.zip")
        self.env.close()

    def evaluate(self):
        # Load the model (if not already in memory)
        if self.model is None:
            self.model = DQN.load("dqn_custom_lunar_lander")
        # Create an environment that can render to the screen
        self.env = self.create_env(render_mode="human")

        obs, info = self.env.reset()
        done = False
        while not done:
            action, _ = self.model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated

        self.env.close()

    class RewardTrackingCallback(BaseCallback):
        """
        A custom callback to track and plot rewards after each episode.
        """

        def __init__(self, verbose=0):
            super().__init__(verbose)
            self.episode_rewards = []
            self.episode_lengths = []
            self.episodes = []
            self.episode_count = 0
            self.last_100_rewards = deque(maxlen=100)

            # File for saving rewards
            self.filepath = "results/lunarlander_rewards.csv"
            with open(self.filepath, "w") as f:
                f.write("Episode,Reward,Length\n")

            # Set up a live plot figure for rewards
            self.fig, self.ax = plt.subplots()
            self.ax.set_xlabel("Episode")
            self.ax.set_ylabel("Reward")
            plt.ion()  # Interactive mode for live updates

        def _on_step(self) -> bool:
            # The 'infos' are from each step, but we only look for the final 'episode' info
            infos = self.locals.get("infos", [])
            for info in infos:
                if "episode" in info:
                    episode_reward = info["episode"]["r"]
                    episode_length = info["episode"]["l"]
                    self.episode_rewards.append(episode_reward)
                    self.episode_lengths.append(episode_length)
                    self.episodes.append(self.episode_count)
                    self.last_100_rewards.append(episode_reward)
                    self.episode_count += 1

                    # Log to CSV file
                    with open(self.filepath, "a") as f:
                        f.write(
                            f"{self.episode_count},{episode_reward},{episode_length}\n"
                        )

                    # Print diagnostic info
                    print(
                        f"Episode {self.episode_count}: Reward={episode_reward:.2f}, Length={episode_length}"
                    )

                    # Update the live plot
                    self.ax.clear()
                    self.ax.set_xlabel("Episode")
                    self.ax.set_ylabel("Reward")
                    self.ax.plot(
                        self.episodes, self.episode_rewards, label="Episode Reward"
                    )
                    self.ax.legend()
                    plt.draw()
                    plt.pause(0.001)

                    # Save the current plot to a PNG file each episode
                    plt.savefig("results/reward_plot.png")

            return True  # Continue training


# ------------------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    train_mode = True  # Set to False to just evaluate the trained model

    agent = LunarLanderAgent(train_mode=train_mode)

    if train_mode:
        agent.train(total_timesteps=100000)
    else:
        agent.evaluate()

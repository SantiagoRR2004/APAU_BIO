import gymnasium as gym
import numpy as np
from gymnasium.envs.box2d import LunarLander
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
import matplotlib.pyplot as plt
from collections import deque
import os

currentDirectory = os.path.dirname(os.path.abspath(__file__))
# Create the models directory if it doesn't exist
os.makedirs(os.path.join(currentDirectory, "models"), exist_ok=True)


class CustomLunarLanderEnv(LunarLander):
    def __init__(self, **kwargs):
        super(CustomLunarLanderEnv, self).__init__(**kwargs)
        # Define the fuel penalty multiplier
        self.fuel_penalty_multiplier = 2.0  # Adjust this value as needed

    def step(self, action):
        # Call the parent class's step function to get the original outputs
        state, reward, terminated, truncated, info = super(
            CustomLunarLanderEnv, self
        ).step(action)
        # Modify the reward here
        reward = self.custom_reward_function(state, reward, action)
        return state, reward, terminated, truncated, info

    def custom_reward_function(self, state, reward, action):
        # Define your custom reward logic
        # Extract fuel usage from the action
        fuel_usage = 0.0
        if action == 1 or action == 3:
            # Side engines consume less fuel
            fuel_usage += 0.03
        elif action == 2:
            # Main engine consumes more fuel
            fuel_usage += 0.3

        # Increase the penalty for fuel usage
        fuel_penalty = self.fuel_penalty_multiplier * fuel_usage

        # Modify the reward
        reward -= fuel_penalty
        return reward


class LunarLanderAgent:
    def __init__(self, train_mode=True):
        self.train_mode = train_mode
        self.env = None
        self.model = None
        self.callback = None
        self.fileName = os.path.join(
            currentDirectory, "models", "dqn_custom_lunar_lander"
        )

        # Create directory to store results if not exists
        os.makedirs(os.path.join(currentDirectory, "results"), exist_ok=True)

    def create_env(self, render_mode=None):
        # Create the custom environment
        env = CustomLunarLanderEnv(render_mode=render_mode)
        # Wrap the environment with Monitor
        env = Monitor(env)
        return env

    def create_model(self):
        # Create the DQN model with TensorBoard logging
        self.model = DQN(
            "MlpPolicy",
            self.env,
            verbose=1,
            tensorboard_log=os.path.join(currentDirectory, "logs"),
        )

    def train(self, total_timesteps=100000):
        # Create the environment
        self.env = self.create_env()
        # Create the model
        self.create_model()
        # Create the callback
        self.callback = self.RewardTrackingCallback()
        # Train the model
        self.model.learn(total_timesteps=total_timesteps, callback=self.callback)
        # Save the model
        self.model.save(self.fileName)
        # Close the environment
        self.env.close()

    def evaluate(self):
        # Load the model
        if self.model is None:
            self.model = DQN.load(self.fileName)
        # Create the environment with rendering
        self.env = self.create_env(render_mode="human")
        obs, info = self.env.reset()
        done = False

        while not done:
            action, _states = self.model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            self.env.render()

        self.env.close()

    class RewardTrackingCallback(BaseCallback):
        def __init__(self, verbose=0):
            super().__init__(verbose)
            self.episode_rewards = []
            self.episode_lengths = []
            self.episodes = []
            self.episode_count = 0
            self.last_100_rewards = deque(maxlen=100)

            # Create directory to store results if not exists
            os.makedirs(os.path.join(currentDirectory, "results"), exist_ok=True)

            # Prepare the file for saving rewards
            self.filepath = os.path.join(
                currentDirectory, "results", "lunarlander_rewards.csv"
            )
            with open(self.filepath, "w") as f:
                f.write("Episode,Reward,Length\n")  # Write header

            # Initialize the plot figure
            self.fig, self.ax = plt.subplots()
            self.ax.set_xlabel("Episode")
            self.ax.set_ylabel("Reward")

        def _on_step(self) -> bool:
            # Access the 'infos' from the local variables
            infos = self.locals.get("infos", [])
            for info in infos:
                if "episode" in info.keys():
                    episode_info = info["episode"]
                    episode_reward = episode_info["r"]
                    episode_length = episode_info["l"]
                    self.episode_rewards.append(episode_reward)
                    self.episode_lengths.append(episode_length)
                    self.episodes.append(self.episode_count)
                    self.last_100_rewards.append(episode_reward)
                    self.episode_count += 1

                    # Save to file
                    with open(self.filepath, "a") as f:
                        f.write(
                            f"{self.episode_count},{episode_reward},{episode_length}\n"
                        )

                    # Print diagnostics
                    print(
                        f"Episode {self.episode_count}: Reward = {episode_reward}, Length = {episode_length}"
                    )

                    # Update plot
                    self.ax.clear()
                    self.ax.set_xlabel("Episode")
                    self.ax.set_ylabel("Reward")
                    self.ax.plot(self.episodes, self.episode_rewards)

                    # Save the plot to a PNG file
                    plt.savefig(
                        os.path.join(currentDirectory, "results", "reward_plot.png")
                    )

            return True


if __name__ == "__main__":
    # Set train_mode to True to train, or False to evaluate
    train_mode = True  # Set to False to run the trained model

    agent = LunarLanderAgent(train_mode=train_mode)

    if train_mode:
        agent.train(total_timesteps=100000)
    else:
        agent.evaluate()

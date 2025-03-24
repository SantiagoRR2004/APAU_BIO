import gymnasium as gym
from stable_baselines3 import PPO
import matplotlib.pyplot as plt
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
import os
from collections import deque

################################################################################
# Choose your environment here by uncommenting the line you want:
################################################################################
env_name = "CartPole-v1"
# env_name = "LunarLander-v2"
# env_name = "MountainCar-v0"
################################################################################

# Define thresholds for “solving” each environment
ENV_THRESHOLDS = {
    "CartPole-v1": 195,       # Often used as a "solved" threshold
    "LunarLander-v2": 200,    # Example threshold
    "MountainCar-v0": -110    # Example threshold
}

# Step 1: Define a custom callback to track rewards and stop when the problem is solved
class RewardTrackingCallback(BaseCallback):
    def __init__(self, env_name, run_number, verbose=0):
        super().__init__(verbose)
        self.env_name = env_name
        self.run_number = run_number

        # Retrieve the threshold from our dictionary
        self.threshold = ENV_THRESHOLDS.get(env_name, 195)  # default=195 if env not listed

        self.episode_rewards = []
        self.episode_lengths = []
        self.episodes = []
        self.current_rewards = []
        self.episode_count = 0

        # Number of consecutive episodes to average for 'solved' condition
        self.solved_episodes = 100  
        self.last_100_rewards = deque(maxlen=self.solved_episodes)

        self.solved = False  # Flag to indicate if environment has been solved

        # ---- Setup live plotting ----
        plt.ion()  # Turn on interactive mode
        self.fig, self.ax = plt.subplots()
        self.ax.set_xlabel("Episode")
        self.ax.set_ylabel("Reward")

        # Create directory to store results if not exists
        if not os.path.exists('results'):
            os.makedirs('results')

        # Prepare the file for saving rewards (with env_name and run_number in the filename)
        self.filepath = f"results/{self.env_name}_rewards_run_{self.run_number}.csv"
        with open(self.filepath, 'w') as f:
            f.write("Episode,Reward,Length\n")  # Write header

    def _on_step(self) -> bool:
        # Retrieve current rewards from the environment
        # For vectorized env, we typically look at self.locals['rewards'][0]
        reward = self.locals['rewards'][0]
        done = self.locals['dones'][0]

        # Accumulate rewards for the current episode
        self.current_rewards.append(reward)

        if done:  # Episode finished
            episode_reward = np.sum(self.current_rewards)
            episode_length = self.num_timesteps  # total timesteps so far

            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(episode_length)
            self.episodes.append(self.episode_count)
            self.last_100_rewards.append(episode_reward)
            self.episode_count += 1

            # Save to CSV
            with open(self.filepath, 'a') as f:
                f.write(f"{self.episode_count},{episode_reward},{episode_length}\n")

            # Print diagnostics
            print(f"Episode {self.episode_count}: "
                  f"Reward = {episode_reward:.2f}, "
                  f"Total Timesteps = {episode_length}")

            # Check if the problem is solved (average over last 100 episodes)
            if len(self.last_100_rewards) == self.solved_episodes:
                mean_score = np.mean(self.last_100_rewards)
                if mean_score >= self.threshold:
                    print(f"\n{self.env_name} solved after {self.episode_count} episodes "
                          f"with average reward: {mean_score:.2f} ✔")
                    self.solved = True
                    return False  # Stop training

            # Reset current episode rewards
            self.current_rewards = []

            # ---- Update live plot ----
            self.ax.clear()
            self.ax.set_title(f"{self.env_name} (Run {self.run_number})")
            self.ax.set_xlabel("Episode")
            self.ax.set_ylabel("Reward")
            self.ax.plot(self.episodes, self.episode_rewards, label="Episode Reward")
            self.ax.legend()
            plt.draw()
            plt.pause(0.001)

        return True  # Continue training

################################################################################
# MAIN: Train or Load
################################################################################
train_mode = True  # Set to True to train a new model, False to load and evaluate

if train_mode:
    # Create environment
    # render_mode="rgb_array": does not open a window, but we can see frames in memory
    # if you prefer a window, use render_mode="human"
    env = gym.make(env_name, render_mode="rgb_array")

    # We can run multiple times if desired
    for run in range(1, 2):  # Adjust range to run multiple training sessions
        print(f"Starting run {run} for {env_name}...")

        # Step 3: Train the model with the custom callback
        callback = RewardTrackingCallback(env_name=env_name, run_number=run)
        model = PPO("MlpPolicy", env, verbose=0)  # Using PPO
        model.learn(total_timesteps=200_000, callback=callback)  # Stop early if solved

        # Step 4: Save the trained model
        model_path = f"results/ppo_{env_name}_run_{run}"
        model.save(model_path)
        print(f"Model saved to {model_path}")

        if callback.solved:
            print(f"Training stopped early for run {run} as the problem was solved.\n")
            break  # Exit the loop if solved
else:
    # Step 5: Load and evaluate the model
    run_number = 1  # Change this to load a specific run
    model_path = f"results/ppo_{env_name}_run_{run_number}"
    print(f"Loading model from: {model_path}")
    loaded_model = PPO.load(model_path)

    # Step 6: Run the environment with the trained model, in a window
    env = gym.make(env_name, render_mode="human")
    obs, _ = env.reset()

    for _ in range(1000):  # Run for a fixed number of steps
        action, _states = loaded_model.predict(obs)
        obs, reward, done, truncated, info = env.step(action)
        # If the environment says the episode is finished, reset
        if done or truncated:
            obs, _ = env.reset()

    env.close()

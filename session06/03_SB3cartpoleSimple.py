import gymnasium as gym
from stable_baselines3 import PPO
import os

# 1. Create the CartPole environment
env = gym.make("CartPole-v1")

# 2. Instantiate the PPO model (policy gradient method)
#    You can adjust hyperparameters as needed (e.g., learning_rate, n_steps, etc.)
model = PPO("MlpPolicy", env, verbose=1)

# 3. Train the model
model.learn(total_timesteps=10000)

# 4. Save the model
currentDirectory = os.path.dirname(os.path.abspath(__file__))
# Create the models directory if it doesn't exist
os.makedirs(os.path.join(currentDirectory, "models"), exist_ok=True)
model.save(os.path.join(currentDirectory, "models", "ppo_cartpole"))
del model  # Remove model from memory

del env  # Remove environment from memory
env = gym.make("CartPole-v1", render_mode="human")

# 5. Load the trained model
model = PPO.load("ppo_cartpole", env=env)

# 6. Evaluate or run the environment with the trained model
obs, _ = env.reset()
total_reward = 0

for _ in range(1000):
    action, _states = model.predict(obs.reshape(1, -1))
    action = int(action.item())  # Convert action to scalar integer
    obs, reward, done, truncated, info = env.step(action)
    total_reward += reward
    env.render()
    if done or truncated:
        obs, _ = env.reset()
        print(f"Total reward: {total_reward}/{env.spec.reward_threshold}")
        total_reward = 0

env.close()

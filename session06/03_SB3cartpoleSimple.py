import gymnasium as gym
from stable_baselines3 import PPO

# 1. Create the CartPole environment
env = gym.make("CartPole-v1")

# 2. Instantiate the PPO model (policy gradient method)
#    You can adjust hyperparameters as needed (e.g., learning_rate, n_steps, etc.)
model = PPO("MlpPolicy", env, verbose=1)

# 3. Train the model
model.learn(total_timesteps=10000)

# 4. Save the model
model.save("ppo_cartpole")
del model  # Remove model from memory

# 5. Load the trained model
model = PPO.load("ppo_cartpole", env=env)

# 6. Evaluate or run the environment with the trained model
obs = env.reset()

for _ in range(1000):
    action, _states = model.predict(obs)
    obs, reward, done, truncated, info = env.step(action)
    env.render()
    if done or truncated:
        obs = env.reset()

env.close()

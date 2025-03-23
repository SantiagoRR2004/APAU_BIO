#!/usr/bin/env python
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import matplotlib.pyplot as plt
import time
import os


class Agent:
    def __init__(
        self,
        env_string: str,
        dqnClass: nn.Module,
        batch_size: int = 64,
        render_mode: str = None,
        update_target_steps: int = 100,
        fileName: str = None,
    ) -> None:
        self.memory = deque(maxlen=100000)
        self.env = gym.make(env_string, render_mode=render_mode)

        if not isinstance(self.env.observation_space, gym.spaces.box.Box):
            print("Not supported: observation space is not continuous.")
            # If they are discrete, we can flatten, but the training needs to be modified
            self.error = True
            return
        else:
            self.error = False
        self.input_size = self.env.observation_space.shape[0]

        if isinstance(self.env.action_space, gym.spaces.Box):
            print("Continuous action space. Not supported.")
            self.error = True
            return
        else:
            self.error = False
        self.action_size = self.env.action_space.n

        self.batch_size = batch_size
        self.gamma = 1.0
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.update_target_steps = update_target_steps

        self.threshold = (
            self.env.spec.reward_threshold
            if self.env.spec.reward_threshold is not None
            else float("inf")
        )

        if self.threshold == float("inf"):
            print(
                f"Warning: No reward threshold found for {env_string}. It will be set to 95."
            )
            self.threshold = 95

        if fileName is None:
            fileName = f"dqn_{env_string}.pth"
        self.fileName = fileName

        # Main network
        self.model = dqnClass(self.input_size, self.action_size).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.004)

        # Target network
        self.target_model = dqnClass(self.input_size, self.action_size).to(self.device)
        self.update_target_network()

        self.steps_done = 0
        self.scores = []
        self.avg_scores = []
        self.losses = []  # <--- Track losses

        if render_mode != "human":
            self.setup_plot()

        return

    def update_target_network(self):
        """Copy weights from main network to target network."""
        self.target_model.load_state_dict(self.model.state_dict())

    def setup_plot(self):
        plt.ion()
        self.fig, self.ax = plt.subplots()
        self.ax.set_xlabel("Episode")
        self.ax.set_ylabel("Mean Reward")
        plt.show()

    def update_plot(self):
        self.ax.clear()
        # Plot the average score
        self.ax.plot(self.avg_scores, label="Avg Reward")
        # Plot the loss on the same axis (optional, may have different scale)
        self.ax.plot(self.losses, label="Loss", color="red")
        self.ax.set_xlabel("Episode")
        self.ax.set_title("Mean Reward and Loss during Training")
        self.ax.legend()
        plt.draw()
        plt.pause(0.001)

    def preprocess_state(self, state):
        if isinstance(state, tuple):
            state = state[0]
        state = np.array(state, dtype=np.float32)
        if state.ndim == 1:
            state = np.expand_dims(state, axis=0)
        return state

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def choose_action(self, state, epsilon):
        if np.random.rand() <= epsilon:
            return self.env.action_space.sample()
        else:
            state = torch.FloatTensor(state).to(self.device)
            with torch.no_grad():
                action_values = self.model(state)
            return torch.argmax(action_values).item()

    def replay(self):
        if len(self.memory) < self.batch_size:
            return None  # Not enough data to train

        minibatch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*minibatch)

        states = torch.FloatTensor(np.vstack(states)).to(self.device)
        next_states = torch.FloatTensor(np.vstack(next_states)).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        current_q_values = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        next_q_values = self.target_model(next_states).max(1)[0]
        expected_q_values = rewards + self.gamma * next_q_values * (1 - dones)

        loss = nn.MSELoss()(current_q_values, expected_q_values.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.steps_done += 1
        if self.steps_done % self.update_target_steps == 0:
            self.update_target_network()

        return loss.item()

    def train_model(self, epochs: int = 10000, threshold: int = None):
        if self.error:
            return
        scores = deque(maxlen=100)

        if threshold is None:
            threshold = self.threshold
            print(f"Using default threshold of {threshold}")

        for epoch in range(epochs):
            state = self.env.reset()
            state = self.preprocess_state(state)
            done = False
            total_reward = 0

            while not done:
                action = self.choose_action(state, self.epsilon)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                next_state = self.preprocess_state(next_state)
                self.remember(state, action, reward, next_state, done)
                state = next_state
                self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
                total_reward += reward

            scores.append(total_reward)
            mean_score = np.mean(scores)
            self.avg_scores.append(mean_score)

            loss_val = None
            if len(self.memory) >= self.batch_size:
                loss_val = self.replay()

            if loss_val is not None:
                self.losses.append(loss_val)

            self.update_plot()

            if mean_score >= threshold and len(scores) == 100:
                print(f"Ran {epoch} episodes. Solved after {epoch - 100} trials ✔")
                torch.save(self.model.state_dict(), self.fileName)
                plt.ioff()
                return self.avg_scores

            if epoch % 100 == 0:
                print(
                    f"[Episode {epoch}] - Mean survival time over last 100 episodes: {mean_score:.2f}"
                )

        print(f"Did not solve after {epochs} episodes")
        torch.save(self.model.state_dict(), self.fileName)
        plt.ioff()
        return self.avg_scores

    def load_weights_and_visualize(self):
        if self.error:
            return
        self.model.load_state_dict(
            torch.load(self.fileName, map_location=torch.device("cpu"))
        )
        self.model.eval()

        for episode in range(5):
            state = self.env.reset()
            state = self.preprocess_state(state)
            done = False
            totalReward = 0
            while not done:
                self.env.render()
                time.sleep(0.05)
                state = torch.FloatTensor(state).to(self.device)
                with torch.no_grad():
                    action_values = self.model(state)
                action = torch.argmax(action_values).item()

                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                next_state = self.preprocess_state(next_state)
                state = next_state
                totalReward += reward

            print(f"Total reward: {totalReward}")

        print("Visualization complete.")
        # input()
        self.env.close()


def trainModel(
    env_string: str,
    dqnClass: nn.Module,
    fileName: str,
    retrain: bool = False,
    threshold: int = None,
    epochs: int = 10000,
) -> None:
    """
    Trains a DQN model on the specified environment.

    Args:
        - env_string (str): The name of the environment to train on.
        - dqnClass (nn.Module): The class that defines the DQN model.
        - fileName (str): The name of the file to save the model weights.
        - retrain (bool): If True, the model will be retrained from scratch.
        - threshold (int): The reward threshold for the environment.
        - epochs (int): The number of episodes to train the model for.

    threshold and epochs are only used if the model is trained.

    Returns:
        - None
    """
    # Check if the file exists
    if os.path.exists(fileName) and not retrain:
        agent = Agent(
            env_string, render_mode="human", dqnClass=dqnClass, fileName=fileName
        )
        agent.load_weights_and_visualize()
    else:
        agent = Agent(
            env_string, render_mode=None, dqnClass=dqnClass, fileName=fileName
        )
        scores = agent.train_model(threshold=threshold, epochs=epochs)

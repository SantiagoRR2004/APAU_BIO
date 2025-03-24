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

class DQN(nn.Module):
    def __init__(self, input_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, 24)
        self.fc2 = nn.Linear(24, 48)
        self.fc3 = nn.Linear(48, action_size)
        self.relu = nn.Tanh()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)  # Linear output (Q-values for each action)

class Agent():
    def __init__(self, env_string, batch_size=64, render_mode=None, update_target_steps=100):
        self.memory = deque(maxlen=100000)
        self.env = gym.make(env_string, render_mode=render_mode)
        self.input_size = self.env.observation_space.shape[0]
        self.action_size = self.env.action_space.n

        self.batch_size = batch_size
        self.gamma = 1.0
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.update_target_steps = update_target_steps

        # Main network
        self.model = DQN(self.input_size, self.action_size).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.004)

        # Target network
        self.target_model = DQN(self.input_size, self.action_size).to(self.device)
        self.update_target_network()

        self.steps_done = 0
        self.scores = []
        self.avg_scores = []
        self.losses = []  # <--- Track losses

        self.setup_plot()

    def update_target_network(self):
        """Copy weights from main network to target network."""
        self.target_model.load_state_dict(self.model.state_dict())

    def setup_plot(self):
        plt.ion()
        self.fig, self.ax = plt.subplots()
        self.ax.set_xlabel('Episode')
        self.ax.set_ylabel('Mean Reward')
        plt.show()

    def update_plot(self):
        self.ax.clear()
        # Plot the average score
        self.ax.plot(self.avg_scores, label='Avg Reward')
        # Plot the loss on the same axis (optional, may have different scale)
        self.ax.plot(self.losses, label='Loss', color='red')
        self.ax.set_xlabel('Episode')
        self.ax.set_title('Mean Reward and Loss during Training')
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

    def train_model(self, epochs=10000, threshold=95):
        scores = deque(maxlen=100)

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
                print(f'Ran {epoch} episodes. Solved after {epoch - 100} trials ✔')
                torch.save(self.model.state_dict(), 'dqn_cartpole.pth')
                plt.ioff()
                return self.avg_scores

            if epoch % 100 == 0:
                print(f'[Episode {epoch}] - Mean survival time over last 100 episodes: {mean_score:.2f}')

        print(f'Did not solve after {epochs} episodes')
        torch.save(self.model.state_dict(), 'dqn_cartpole.pth')
        plt.ioff()
        return self.avg_scores

    def load_weights_and_visualize(self):
        self.model.load_state_dict(torch.load('dqn_cartpole.pth'))
        self.model.eval()

        for episode in range(5):
            state = self.env.reset()
            state = self.preprocess_state(state)
            done = False
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

        print("Visualization complete. Press Enter to close.")
        input()
        self.env.close()


# Usage
train_mode = True  # Change to True if you want to retrain
if train_mode:
    agent = Agent('CartPole-v1', render_mode=None)
    scores = agent.train_model()
else:
    agent = Agent('CartPole-v1', render_mode='human')
    agent.load_weights_and_visualize()


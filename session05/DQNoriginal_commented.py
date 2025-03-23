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


class Agent:
    def __init__(
        self, env_string, batch_size=64, render_mode=None, update_target_steps=100
    ):
        self.memory = deque(maxlen=100000)
        self.env = gym.make(env_string, render_mode=render_mode)
        # The number of input features (4 for CartPole): position, cart velocity, pole angle, pole angular velocity
        self.input_size = self.env.observation_space.shape[0]
        print("input_size:", self.input_size)
        print(
            "observation_space:", self.env.observation_space
        )  # Box(4,) means 4 continuous values
        self.action_size = self.env.action_space.n
        print("action_size:", self.action_size)  # two actions: 0 (left) and 1 (right)
        print(
            "action_space:", self.env.action_space
        )  # Discrete(2) means 2 discrete actions

        self.batch_size = batch_size  # number of samples drawn from memory (replay buffer) to train the model
        self.gamma = 1.0  # Discount factor, 1: consider future rewards, 0: consider only immediate reward
        # Exploration rate (1: explore, 0: exploit), prob of choosing random action or optimal action
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.update_target_steps = (
            update_target_steps  # How often to update the target network
        )

        # Main network (used for action selection and learning)
        self.model: DQN = DQN(self.input_size, self.action_size).to(self.device)
        self.optimizer = optim.Adam(
            self.model.parameters(), lr=0.004
        )  # FIXME: learning rate

        # Target network (used to compute target Q-values)
        self.target_model = DQN(self.input_size, self.action_size).to(self.device)
        self.update_target_network()  # Initialize the target network with the same weights as the main network

        self.steps_done = (
            0  # Keep track of steps to decide when to update the target network
        )

        self.scores = []
        self.avg_scores = []
        self.setup_plot()

    def update_target_network(self):
        """Copy the weights from the main network to the target network."""
        self.target_model.load_state_dict(self.model.state_dict())

    def setup_plot(self):
        plt.ion()
        self.fig, self.ax = plt.subplots()
        self.ax.set_xlabel("Episode")
        self.ax.set_ylabel("Mean Reward")
        plt.show()

    def update_plot(self):
        self.ax.clear()
        self.ax.plot(self.avg_scores)
        self.ax.set_xlabel("Episode")
        self.ax.set_ylabel("Mean Reward")
        self.ax.set_title("Mean Reward during Training")
        plt.draw()
        plt.pause(0.001)

    def preprocess_state(self, state):
        """Convert the state to a observation tensor.
        This is needed to convert the state to a tensor before passing it to the model.
        """
        if isinstance(state, tuple):
            state = state[0]
        state = np.array(state, dtype=np.float32)
        if state.ndim == 1:
            state = np.expand_dims(state, axis=0)
        return state

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def choose_action(self, state, epsilon):
        """Epsilon greedy choice of action."""
        if np.random.rand() <= epsilon:
            return (
                self.env.action_space.sample()
            )  # Random action with probability epsilon
        else:
            # a = argmax_a Q_theta(s,a) with probability 1-epsilon
            state = torch.FloatTensor(state).to(self.device)
            with torch.no_grad():
                action_values = self.model(state)
            return torch.argmax(
                action_values
            ).item()  # Return the action with highest Q-value

    def replay(self):
        if len(self.memory) < self.batch_size:
            return 0  # Not enough data to train yet

        # Sample a minibatch from memory
        minibatch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*minibatch)

        # Convert to appropriate tensors
        states = torch.FloatTensor(np.vstack(states)).to(
            self.device
        )  # stack the states vertically
        next_states = torch.FloatTensor(np.vstack(next_states)).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # Get current Q-values for actions taken (from main network)
        # self.model(states) = forward pass of the model, returns Q-values for all action-state pairs
        # gather(1, actions.unsqueeze(1)) = select the Q-values for the actions taken
        #   actions.unsqueeze(1) = convert actions to a column vector aka transpose
        #   gather(1, ...) = select the Q-values for the actions taken
        # squeeze(1) = remove the extra dimension added by unsqueeze (back to 1D tensor)
        current_q_values = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Get max predicted Q-values for next states from the target network
        next_q_values = self.target_model(next_states).max(1)[0]

        # Compute expected Q-values using the discount factor + don't add if state is terminal
        # Q(s,a) = r + gamma * max_a' Q(s',a') if s' is not terminal else r
        expected_q_values = rewards + self.gamma * next_q_values * (1 - dones)

        # Calculate loss (Mean Squared Error)
        # loss = 1/K * sum_i (Q_theta_target(s,a) - Q_theta_main(s,a))^2
        # loss = 1/K * sum_i (expected_q_values - current_q_values)^2
        # loss = 1/K * sum_i (r + gamma * max_a' Q(s',a') - Q(s,a))^2
        loss = nn.MSELoss()(current_q_values, expected_q_values.detach())

        # Backpropagation
        # with this, after training the main network and propagating the loss,
        # the Q-values will be closer to the target
        # then we update the target network every few steps as to not create instability in the training
        self.optimizer.zero_grad()  # Zero the gradients
        loss.backward()  # Compute the gradients
        self.optimizer.step()  # Update the weights AKA parameters

        # Update target network weights every few steps
        self.steps_done += 1
        if self.steps_done % self.update_target_steps == 0:
            self.update_target_network()

        return loss.item()

    def train_model(
        self, epochs=10000, threshold=99  # epochs aka episodes aka trajectories
    ):  # FIXME: normalize for different environments

        scores = deque(maxlen=100)  # FIXME : why?

        for epoch in range(epochs):
            state = self.env.reset()  # Reset the environment
            state = self.preprocess_state(state)
            done = False
            total_reward = 0

            while not done:
                action = self.choose_action(state, self.epsilon)
                # go from state to next_state
                # outputs : next state obsevation, float reward, end state flag, out of bounds / time flag
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = (
                    terminated or truncated
                )  # end of trajectory condition, stop doing actions
                next_state = self.preprocess_state(next_state)
                self.remember(state, action, reward, next_state, done)
                state = next_state
                # gradually reduce exploration rate and increase exploitation rate
                self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
                total_reward += reward

            scores.append(total_reward)  # save some data
            mean_score = np.mean(scores)
            self.avg_scores.append(mean_score)

            # here we do the importans stuff: compute the loss and backpropagate to update the weights
            if len(self.memory) >= self.batch_size:
                self.replay()

            self.update_plot()

            if mean_score >= threshold and len(scores) == 100:
                print(f"Ran {epoch} episodes. Solved after {epoch - 100} trials ✔")
                torch.save(self.model.state_dict(), "dqn_cartpole.pth")
                plt.ioff()
                return self.avg_scores

            # TODO save checkpoints of best models instead of saving the last one or the one that reached the threshold

            if epoch % 100 == 0:
                print(
                    f"[Episode {epoch}] - Mean survival time over last 100 episodes was {mean_score} ticks."
                )

        print(f"Did not solve after {epochs} episodes")
        torch.save(self.model.state_dict(), "dqn_cartpole.pth")
        plt.ioff()
        return self.avg_scores

    def load_weights_and_visualize(self):
        self.model.load_state_dict(torch.load("dqn_cartpole.pth"))
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
                    action_values = self.model(state)  # forward pass to get Q-values
                # select the action with the highest Q-value
                action = torch.argmax(action_values).item()

                # perform the action and go to the next state
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                next_state = self.preprocess_state(next_state)
                state = next_state

        print("Visualization complete. Press Enter to close the window.")
        input()
        self.env.close()


# Usage
train_mode = False

if train_mode:
    agent = Agent("CartPole-v1", render_mode=None)
    scores = agent.train_model()
else:
    agent = Agent("CartPole-v1", render_mode="human")
    agent.load_weights_and_visualize()

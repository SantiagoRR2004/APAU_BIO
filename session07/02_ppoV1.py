import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# 1) Actor (Policy) and Critic (Value) Networks
# ------------------------------------------------------------
class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        return self.fc(x)


class ValueNetwork(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.fc(x).squeeze(-1)


# ------------------------------------------------------------
# 2) PPO Clipped Objective
# ------------------------------------------------------------
def ppo_clip_loss(policy, states, actions, old_log_probs, advantages, eps_clip=0.2):
    """
    Computes the clipped PPO objective for the policy:
        ratio = exp(new_log_prob - old_log_prob)
        surr1 = ratio * advantage
        surr2 = clip(ratio, 1 - eps, 1 + eps) * advantage
        policy_loss = - E[ min(surr1, surr2) ]
    """
    # Compute new log-prob of the actions under current policy
    probs = policy(states).clamp(min=1e-8)
    new_log_probs = torch.log(probs.gather(1, actions.unsqueeze(-1)).squeeze(-1))

    ratio = torch.exp(new_log_probs - old_log_probs)  # shape (T,)
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1 - eps_clip, 1 + eps_clip) * advantages

    policy_loss = -torch.mean(torch.min(surr1, surr2))  # negative for gradient ASCENT
    return policy_loss


# ------------------------------------------------------------
# 3) GAE (Generalized Advantage Estimation)
# ------------------------------------------------------------
def compute_returns_and_advantages(rewards, values, gamma=0.99, lam=0.95):
    """
    Given a single (or batched) trajectory of rewards and value predictions:
      - Compute 'returns' (discounted sum of future rewards)
      - Compute GAE advantages using 'values' and 'values_next'
    """
    T = len(rewards)
    returns = [0] * T
    advantages = [0] * T

    # 1) Compute returns (rewards-to-go)
    G = 0
    for t in reversed(range(T)):
        G = rewards[t] + gamma * G
        returns[t] = G

    # 2) Compute advantages via GAE
    #    advantage_t = delta_t + gamma*lam*delta_{t+1} + ...
    #    where delta_t = r_t + gamma*V(s_{t+1}) - V(s_t)
    A = 0
    for t in reversed(range(T)):
        if t == T - 1:
            values_next = 0.0
        else:
            values_next = values[t+1].item()
        delta = rewards[t] + gamma * values_next - values[t].item()
        A = delta + gamma * lam * A
        advantages[t] = A

    returns = torch.tensor(returns, dtype=torch.float32)
    advantages = torch.tensor(advantages, dtype=torch.float32)
    return returns, advantages


# ------------------------------------------------------------
# 4) The Main PPO Training Loop
# ------------------------------------------------------------
def train_ppo(
    env,
    policy,
    value_net,
    max_episodes=500,
    gamma=0.99,
    lam=0.95,
    ppo_clip=0.2,
    epochs=10,
    lr_policy=3e-4,
    lr_value=1e-3
):
    """
    Collects one entire episode of data each iteration, then performs
    multiple epochs of PPO updates on that data, for demonstration.

    - env: a Gymnasium environment
    - policy: policy network (actor)
    - value_net: critic network
    - max_episodes: how many episodes to run
    - ppo_clip: epsilon in the clipped PPO objective
    - epochs: how many epochs of gradient updates to run each iteration
    - lr_policy, lr_value: learning rates
    """
    # Separate optimizers for policy and value networks
    policy_optimizer = optim.Adam(policy.parameters(), lr=lr_policy)
    value_optimizer = optim.Adam(value_net.parameters(), lr=lr_value)

    all_returns = []  # store episodic returns for plotting

    for episode in range(max_episodes):
        # -----------------------------
        # (A) Collect an episode
        # -----------------------------
        obs, info = env.reset()
        done = False

        states = []
        actions = []
        rewards = []
        log_probs_old = []
        values = []

        while not done:
            # Convert to float32 tensor
            obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)

            # 1) get policy distribution
            with torch.no_grad():
                probs = policy(obs_t).clamp(min=1e-8)
                dist = Categorical(probs)
                action = dist.sample()  # shape ()
                log_prob_action = torch.log(probs[0, action])

                # 2) get value estimate
                value_est = value_net(obs_t)

            next_obs, reward, terminated, truncated, info = env.step(action.item())
            done = terminated or truncated

            # store transitions
            states.append(obs)
            actions.append(action.item())
            rewards.append(reward)
            log_probs_old.append(log_prob_action.item())
            values.append(value_est.item())

            obs = next_obs

        # Compute total episode return
        ep_return = sum(rewards)
        all_returns.append(ep_return)

        # Convert to PyTorch Tensors
        states_t = torch.tensor(states, dtype=torch.float32)
        actions_t = torch.tensor(actions, dtype=torch.int64)
        rewards_t = torch.tensor(rewards, dtype=torch.float32)
        old_log_probs_t = torch.tensor(log_probs_old, dtype=torch.float32)
        values_t = torch.tensor(values, dtype=torch.float32)

        # -----------------------------
        # (B) Compute returns & advantages
        # -----------------------------
        returns_t, advantages_t = compute_returns_and_advantages(
            rewards_t, values_t, gamma=gamma, lam=lam
        )

        # Normalize advantages (common trick in PPO)
        advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)

        # -----------------------------
        # (C) PPO Updates (multiple epochs)
        #     Here we do a single-batch update (the entire episode)
        #     For more advanced usage, you'd do minibatches.
        # -----------------------------
        for _ in range(epochs):
            # 1) Policy update (clipped objective)
            policy_loss = ppo_clip_loss(
                policy, states_t, actions_t, old_log_probs_t, advantages_t, eps_clip=ppo_clip
            )
            policy_optimizer.zero_grad()
            policy_loss.backward()
            policy_optimizer.step()

            # 2) Value update (MSE)
            value_estimates = value_net(states_t).squeeze(-1)
            value_loss = torch.mean((returns_t - value_estimates) ** 2)
            value_optimizer.zero_grad()
            value_loss.backward()
            value_optimizer.step()

        # Print result
        print(f"Episode {episode+1}/{max_episodes} - Return: {ep_return:.2f}")

    # -----------------------------
    # Plot the episodic returns
    # -----------------------------
    plt.plot(all_returns)
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.title("PPO with Clipping on CartPole-v1 (Gymnasium)")
    plt.show()


# ------------------------------------------------------------
# 5) Main Entry Point
# ------------------------------------------------------------
if __name__ == "__main__":
    # Use Gymnasium's CartPole
    env = gym.make("CartPole-v1")
    state_dim = env.observation_space.shape[0]  # 4 for CartPole
    action_dim = env.action_space.n            # 2 for CartPole

    # Instantiate policy & value networks
    policy = PolicyNetwork(state_dim, action_dim)
    value_net = ValueNetwork(state_dim)

    # Train PPO
    train_ppo(env, policy, value_net, max_episodes=500)

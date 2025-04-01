import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
import matplotlib.pyplot as plt


# -----------------
# Policy Network
# -----------------
class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.Softmax(dim=-1),
        )

    def forward(self, x):
        return self.fc(x)


# -----------------
# Value Network
# -----------------
class ValueNetwork(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(state_dim, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, x):
        return self.fc(x).squeeze(-1)


# -----------------
# Flatten/Unflatten Helpers
# -----------------
def flat_params(model):
    """Return a flattened view of the model's parameters as a single vector."""
    return torch.cat([p.data.view(-1) for p in model.parameters()])


def set_params(model, new_params):
    """Copy a flat parameter vector back into the model parameters."""
    idx = 0
    for p in model.parameters():
        size = p.numel()
        p.data.copy_(new_params[idx : idx + size].view(p.size()))
        idx += size


# -----------------
# Conjugate Gradient
# -----------------
def conjugate_gradient(Ax, b, max_iter=10, tol=1e-10):
    """
    Solve Ax = b using the conjugate gradient method,
    where Ax is a function that returns A @ x.
    """
    x = torch.zeros_like(b)
    r = b.clone()
    p = r.clone()
    rsold = torch.dot(r, r)

    for _ in range(max_iter):
        Ap = Ax(p)
        alpha = rsold / torch.dot(p, Ap)
        x += alpha * p
        r -= alpha * Ap
        rsnew = torch.dot(r, r)
        if torch.sqrt(rsnew) < tol:
            break
        p = r + (rsnew / rsold) * p
        rsold = rsnew

    return x


# -----------------
# Surrogate Loss
# -----------------
def surrogate_loss(policy, states, actions, advantages, old_log_probs):
    """
    Standard policy gradient surrogate:
       L = E[ advantages * exp(log_pi(a|s) - old_log_pi(a|s)) ]
    """
    # new_probs shape: (T, action_dim)
    new_probs = policy(states).clamp(min=1e-8)
    # gather log-prob of the chosen actions
    new_log_probs = torch.log(new_probs.gather(1, actions.unsqueeze(-1)).squeeze(-1))
    # Surrogate objective
    return (advantages * torch.exp(new_log_probs - old_log_probs)).mean()


# -----------------
# KL Divergence
# -----------------
def kl_divergence(old_probs, new_probs):
    """
    KL(old || new) for each state in the batch, then average.
    old_probs and new_probs are both (T, action_dim), already clamped to avoid log(0).
    KL = sum_i old_probs[i] * log(old_probs[i]/new_probs[i]), then average over batch.
    """
    return (old_probs * (torch.log(old_probs) - torch.log(new_probs))).sum(dim=1).mean()


# -----------------
# TRPO Training
# -----------------
def train_trpo(
    env, policy, value_net, max_episodes=500, gamma=0.99, lam=0.95, max_kl=0.01
):
    """
    Uses the new Gymnasium API, includes plotting, and carefully constrains the KL.
    """
    optimizer = optim.Adam(value_net.parameters(), lr=1e-3)
    all_returns = []  # track returns for plotting

    for episode in range(max_episodes):
        # -------------------------------------
        # 1) Collect one entire episode
        # -------------------------------------
        obs, info = env.reset()
        done = False

        states = []
        actions = []
        rewards = []

        while not done:
            # Convert obs to float32
            obs_np = np.array(obs, dtype=np.float32)  # shape (4,) for CartPole
            obs_tensor = torch.from_numpy(obs_np).unsqueeze(0)

            with torch.no_grad():
                probs = policy(obs_tensor).clamp(min=1e-8)
            dist = Categorical(probs)
            action = dist.sample().item()

            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            states.append(obs_np)
            actions.append(action)
            rewards.append(reward)

            obs = next_obs

        # Convert collected lists to tensors
        states_t = torch.from_numpy(
            np.array(states, dtype=np.float32)
        )  # shape (T, state_dim)
        actions_t = torch.from_numpy(np.array(actions, dtype=np.int64))  # shape (T,)
        rewards_t = torch.from_numpy(np.array(rewards, dtype=np.float32))  # shape (T,)

        # -------------------------------------
        # 2) Compute returns (rewards-to-go)
        # -------------------------------------
        returns = []
        G = 0
        for r in reversed(rewards_t):
            G = r + gamma * G
            returns.insert(0, G)
        returns_t = torch.tensor(returns, dtype=torch.float32)

        # -------------------------------------
        # 3) Compute advantages (GAE)
        # -------------------------------------
        values = value_net(states_t)  # shape (T,)
        values_next = torch.cat([values[1:], torch.tensor([0.0])])
        deltas = rewards_t + gamma * values_next - values

        advantages_list = []
        A = 0
        for delta in reversed(deltas):
            A = delta + gamma * lam * A
            advantages_list.insert(0, A)
        advantages_t = torch.tensor(advantages_list, dtype=torch.float32)

        # -------------------------------------
        # 4) Store old distribution (detached)
        # -------------------------------------
        with torch.no_grad():
            old_probs = policy(states_t).clamp(min=1e-8)
            old_log_probs = torch.log(
                old_probs.gather(1, actions_t.unsqueeze(-1)).squeeze(-1)
            )

        # -------------------------------------
        # 5) Compute gradient of the surrogate
        # -------------------------------------
        surr = surrogate_loss(policy, states_t, actions_t, advantages_t, old_log_probs)
        grad = torch.autograd.grad(surr, policy.parameters(), retain_graph=True)
        grad = torch.cat([g.view(-1) for g in grad])

        # -------------------------------------
        # 6) Build Fisher-vector product
        # -------------------------------------
        def fisher_vector_product(v):
            """
            Re-run forward pass to get new_probs. Then compute:
              KL(old || new) = sum_i old_probs[i]*(log old_probs[i] - log new_probs[i])
            and do one backward pass w.r.t. the policy parameters to find Hessian@v.
            """
            new_probs = policy(states_t).clamp(min=1e-8)
            kl = kl_divergence(old_probs, new_probs)
            grads_kl = torch.autograd.grad(kl, policy.parameters(), create_graph=True)
            flat_grads_kl = torch.cat([g.view(-1) for g in grads_kl])

            grad_v = torch.dot(flat_grads_kl, v)
            grads_v = torch.autograd.grad(grad_v, policy.parameters())
            return torch.cat([g.contiguous().view(-1) for g in grads_v])

        # -------------------------------------
        # 7) Conjugate gradient to find step_dir
        # -------------------------------------
        step_dir = conjugate_gradient(fisher_vector_product, grad)

        # Scale step_dir so that the leading term of KL ~ max_kl
        # The standard TRPO derivation: step_size = sqrt(2*max_kl / (step_dir^T F step_dir)).
        # We'll approximate that by computing step_dir^T F step_dir directly:
        fvp_step_dir = fisher_vector_product(step_dir)  # F d
        # dot(step_dir, F d)
        denom = 0.5 * torch.dot(step_dir, fvp_step_dir)
        if denom.item() < 1e-8:
            # If denom is extremely small, the update could blow up, so skip or reduce step
            print("Warning: denominator in TRPO update is tiny. Skipping update.")
            # We'll just skip the update
            continue
        step_scale = torch.sqrt((2.0 * max_kl) / denom)
        step_dir = step_dir * step_scale

        # -------------------------------------
        # 8) Line search
        #    - revert if KL > max_kl or probs are NaN
        # -------------------------------------
        old_params = flat_params(policy)
        alpha = 1.0
        for _ in range(10):
            new_params = old_params + alpha * step_dir
            set_params(policy, new_params)

            with torch.no_grad():
                # Compute actual KL
                new_probs = policy(states_t).clamp(min=1e-8)
                actual_kl = kl_divergence(old_probs, new_probs)
                # Surrogate
                new_surr = surrogate_loss(
                    policy, states_t, actions_t, advantages_t, old_log_probs
                )

            # Check conditions
            if (
                torch.isnan(new_probs).any()
                or torch.isnan(actual_kl)
                or torch.isnan(new_surr)
            ):
                # Something blew up -> reduce alpha
                alpha *= 0.5
                continue

            if actual_kl > max_kl:
                # Too big a step -> reduce alpha
                alpha *= 0.5
            else:
                # KL is within limit; accept if the surrogate isn't negative
                # (one heuristic; you might also compare it vs old surrogate)
                if new_surr > 0.0:
                    # Accept
                    break
                else:
                    alpha *= 0.5
        else:
            # If we never broke out of the loop, revert to old parameters
            print("Line search failed; reverting to old parameters.")
            set_params(policy, old_params)

        # -------------------------------------
        # 9) Update value network
        # -------------------------------------
        value_loss = (returns_t - value_net(states_t)).pow(2).mean()
        optimizer.zero_grad()
        value_loss.backward()
        optimizer.step()

        # -------------------------------------
        # 10) Logging
        # -------------------------------------
        total_return = returns_t.sum().item()
        all_returns.append(total_return)
        print(f"Episode {episode+1}/{max_episodes}, Return: {total_return:.2f}")

    # -------------------------------------
    # Plot returns
    # -------------------------------------
    plt.plot(all_returns)
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.title("TRPO on CartPole-v1 (Gymnasium)")
    plt.show()


# -----------------
# Main
# -----------------
if __name__ == "__main__":
    # Make sure you're using Gymnasium, e.g. `pip install gymnasium`
    env = gym.make("CartPole-v1")  # Gymnasium environment
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    policy = PolicyNetwork(state_dim, action_dim)
    value_net = ValueNetwork(state_dim)

    train_trpo(env, policy, value_net, max_episodes=500, max_kl=0.01)

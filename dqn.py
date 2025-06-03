
import os
import gym
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from collections import deque
from tqdm import trange, tqdm
import matplotlib.pyplot as plt

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
path = "figures & dqn2"
# os.makedirs(path, exist_ok=True)
# os.makedirs(os.path.join(path,"checkpoints"))


# Replay Buffer
class ReplayBuffer:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        idxs = np.random.choice(len(self.memory), batch_size, replace=False)
        batch = [self.memory[i] for i in idxs]
        s, a, r, ns, d = zip(*batch)
        return np.array(s), np.array(a), np.array(r), np.array(ns), np.array(d)

    def __len__(self):
        return len(self.memory)

class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim)
        )

    def forward(self, x):
        return self.net(x)

# Build discrete action set
def build_actions(n_bins=3):
    bins = np.linspace(-1, 1, n_bins)
    mesh = np.array(np.meshgrid(*[bins] * 4)).T
    return [a for a in mesh.reshape(-1, 4)]

# Epsilon-greedy action selection
def select_action(state, policy_net, n_actions, steps_done,
                  eps_start=1.0, eps_end=0.05, eps_decay=100000):
    eps = eps_end + (eps_start - eps_end) * np.exp(-1. * steps_done / eps_decay)
    if np.random.rand() < eps:
        return np.random.randint(n_actions)
    state_t = torch.from_numpy(state).float().unsqueeze(0).to(device)
    with torch.no_grad():
        q_vals = policy_net(state_t)
    return int(q_vals.argmax(dim=1).item())

# Single-step optimization
def optimize_model(buffer, policy_net, optimizer, batch_size, gamma):
    if len(buffer) < batch_size:
        return
    s, a, r, ns, d = buffer.sample(batch_size)
    s_t = torch.FloatTensor(s).to(device)
    a_t = torch.LongTensor(a).unsqueeze(1).to(device)
    r_t = torch.FloatTensor(r).to(device)
    ns_t = torch.FloatTensor(ns).to(device)
    d_t = torch.FloatTensor(d).to(device)

    # Current Q-values
    q_values = policy_net(s_t).gather(1, a_t).squeeze(1)
    # Next Q-values (no target network)
    next_q = policy_net(ns_t).max(1)[0].detach()
    # Compute target
    target = r_t + gamma * next_q * (1 - d_t)

    loss = nn.MSELoss()(q_values, target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Plotting function with larger figure size
def plot_rewards(rewards, save_path):
    plt.figure(figsize=(12, 8))
    plt.plot(rewards, label='Total Reward')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('BipedalWalker-v3 Training Progress')
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# Training loop
def train(env_name='BipedalWalker-v3', num_episodes=5000,
          batch_size=256, gamma=0.99, lr=1e-4,
          buffer_capacity=32768, n_bins=3, save_dir='figures & dqn'):
    os.makedirs(save_dir, exist_ok=True)
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    actions = build_actions(n_bins)
    n_actions = len(actions)

    policy_net = DQN(state_dim, n_actions).to(device)
    optimizer = optim.Adam(policy_net.parameters(), lr=lr)
    buffer = ReplayBuffer(buffer_capacity)
    episode_rewards = []

    steps_done = 0
    for episode in trange(1, num_episodes + 1, desc='Episodes'):
        reset_out = env.reset()
        state = reset_out[0] if isinstance(reset_out, tuple) else reset_out
        total_reward = 0.0

        for t in range(1, 1601):
            a_idx = select_action(state, policy_net, n_actions, steps_done)
            steps_done += 1
            action = actions[a_idx]
            step_out = env.step(action)
            if len(step_out) == 5:
                next_state, reward, terminated, truncated, _ = step_out
                done = terminated or truncated
            else:
                next_state, reward, done, _ = step_out

            buffer.push(state, a_idx, reward, next_state, done)
            state = next_state
            total_reward += reward

            optimize_model(buffer, policy_net, optimizer, batch_size, gamma)
            if done:
                break

        episode_rewards.append(total_reward)

        if episode % 100 == 0:
            save_path = os.path.join(save_dir, f'reward_plot_episode_{episode:04d}.png')
            plot_rewards(episode_rewards, save_path)
            checkpoint_path = f"figures & dqn2/checkpoints/policy_net_episode_{episode:04d}.pt"
            torch.save(policy_net.state_dict(), checkpoint_path)
        print(f"Episode {episode:4d} — Total Reward: {total_reward:.2f}")

    env.close()

# Entry point
if __name__ == '__main__':
    print(f"Using device: {device}")
    train()

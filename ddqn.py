import os
import gym
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from collections import deque
from tqdm import trange, tqdm
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
# path = "figures & ddqn2"
# os.makedirs(path, exist_ok=True)
# os.makedirs(os.path.join(path,"checkpoints"))

n_bins = 3
bins = np.linspace(-1, 1, n_bins)
actions = [np.array(a) for a in np.array(np.meshgrid(*([bins] * 4))).T.reshape(-1, 4)]
n_actions = len(actions)

# 2. Q-network definition
class DQN(nn.Module):
    def __init__(self, state_dim, n_actions):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, n_actions),
        )

    def forward(self, x):
        return self.net(x)

if __name__ == "__main__":
    # 3. Environment and networks
    env = gym.make("BipedalWalker-v3")
    state_dim = env.observation_space.shape[0]

    policy_net = DQN(state_dim, n_actions).to(device)
    target_net = DQN(state_dim, n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=1e-4)

    # 4. Replay buffer and hyperparameters
    memory = deque(maxlen=32768)
    batch_size = 256
    gamma = 0.99
    eps_start, eps_end, eps_decay = 1.0, 0.05, 100_000

    def select_action(state, steps_done):
        """Epsilon-greedy action selection."""
        eps = eps_end + (eps_start - eps_end) * np.exp(-1. * steps_done / eps_decay)
        if np.random.rand() < eps:
            return np.random.randint(n_actions)
        state_t = torch.from_numpy(state).float().unsqueeze(0).to(device)
        with torch.no_grad():
            qvals = policy_net(state_t)
        return int(qvals.argmax(dim=1).item())

    # 5. Training loop
    num_episodes = 5000
    episode_rewards = []

    for episode in trange(num_episodes, desc="Episodes"):
        # Handle new Gym API for reset()
        reset_out = env.reset()
        if isinstance(reset_out, tuple) and len(reset_out) == 2:
            state, _ = reset_out
        else:
            state = reset_out

        total_reward = 0.0

        for t in tqdm(range(1, 1601), desc="Steps", leave=False):
            a_idx = select_action(state, episode * t)
            action = actions[a_idx]

            # Handle new Gym API for step()
            step_out = env.step(action)
            if len(step_out) == 5:
                next_s, r, terminated, truncated, _ = step_out
                done = terminated or truncated
            else:
                next_s, r, done, _ = step_out

            memory.append((state, a_idx, r, next_s, done))
            state = next_s
            total_reward += r

            if len(memory) >= batch_size:
                batch_idxs = np.random.choice(len(memory), batch_size, replace=False)
                batch = [memory[idx] for idx in batch_idxs]
                s_b, a_b, r_b, ns_b, d_b = zip(*batch)

                # Convert to tensors
                s_b  = torch.tensor(s_b,  dtype=torch.float32, device=device)
                a_b  = torch.tensor(a_b,  dtype=torch.int64,   device=device)
                r_b  = torch.tensor(r_b,  dtype=torch.float32, device=device)
                ns_b = torch.tensor(ns_b, dtype=torch.float32, device=device)
                d_b  = torch.tensor(d_b,  dtype=torch.float32, device=device)

                # Current Q-values
                q_vals = policy_net(s_b).gather(1, a_b.unsqueeze(1)).squeeze(1)
                # Next Q-values from target network
                next_q = target_net(ns_b).max(1)[0].detach()
                # Compute targets
                target = r_b + gamma * next_q * (1 - d_b)

                # Optimize
                loss = nn.MSELoss()(q_vals, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if done:
                break

        episode_rewards.append(total_reward)

        # Periodically update target network
        if episode % 10 == 0:
            target_net.load_state_dict(policy_net.state_dict())

        # # Save reward plot every 100 episodes
        # if episode % 100 == 0 and episode > 0:
        #     plt.figure()
        #     plt.plot(episode_rewards, label="Total Reward")
        #     plt.xlabel("Episode")
        #     plt.ylabel("Total Reward")
        #     plt.title(f"BipedalWalker-v3 — up to Episode {episode}")
        #     plt.legend()
        #     plt.tight_layout()
        #     plt.savefig(f"{path}/reward_plot_episode_{episode:04d}.png")
        #     plt.close()
        #     checkpoint_path = f"{path}/checkpoints/policy_net_episode_{episode:04d}.pt"
        #     torch.save(policy_net.state_dict(), checkpoint_path)
        # print(f"Episode {episode:4d} — Total Reward: {total_reward:.2f}")


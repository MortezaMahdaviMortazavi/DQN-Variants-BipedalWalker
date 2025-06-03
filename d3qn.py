import os
import gym
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from collections import deque, namedtuple
from tqdm import trange, tqdm
import matplotlib.pyplot as plt
import random
import math

# Named tuple for storing transitions
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

# Enhanced Configuration
class Config:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_bins = 3
        self.memory_size = 65536  # Increased buffer size
        self.batch_size = 2048  # Reduced batch size for stability
        self.gamma = 0.995  # Slightly increased discount factor
        self.eps_start = 1.0
        self.eps_end = 0.01  # Lower final epsilon
        self.eps_decay = 150_000  # Slower decay
        self.lr_start = 3e-4  # Dynamic learning rate
        self.lr_end = 1e-5
        self.lr_decay_start = 700  # Start decaying after episode 700
        self.target_update_freq = 10
        self.soft_update_tau = 0.005  # Soft target updates
        self.plot_freq = 100
        self.num_episodes = 6000
        self.max_steps = 1600
        self.figure_size = (12, 8)
        self.figure_dpi = 300
        self.path = "figures & d3qn-enhanced2"
        
        # PER parameters
        self.per_alpha = 0.6  # Prioritization exponent
        self.per_beta_start = 0.4  # Importance sampling correction
        self.per_beta_end = 1.0
        self.per_epsilon = 1e-6  # Small positive constant
        
        # Gradient clipping and regularization
        self.grad_clip = 1.0
        self.weight_decay = 1e-5
        
        # Early stopping and performance monitoring
        self.patience = 500
        self.min_improvement = 10.0
        self.performance_window = 100

# Action space setup (same as original)
class ActionSpace:
    def __init__(self, n_bins=3):
        self.n_bins = n_bins
        self.bins = np.linspace(-1, 1, n_bins)
        self.actions = [np.array(a) for a in np.array(np.meshgrid(*([self.bins] * 4))).T.reshape(-1, 4)]
        self.n_actions = len(self.actions)

# Enhanced Dueling DQN with Noisy Layers and Better Initialization
class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, std_init=0.1):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init
        
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))
        
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer('bias_epsilon', torch.empty(out_features))
        
        self.reset_parameters()
        self.reset_noise()
    
    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))
    
    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        # Use detach() to avoid gradient tracking issues
        self.weight_epsilon.data.copy_(epsilon_out.ger(epsilon_in).detach())
        self.bias_epsilon.data.copy_(epsilon_out.detach())
    
    def _scale_noise(self, size):
        x = torch.randn(size, device=self.weight_mu.device)
        return x.sign().mul_(x.abs().sqrt_())
    
    def forward(self, input):
        if self.training:
            return nn.functional.linear(input, self.weight_mu + self.weight_sigma * self.weight_epsilon, 
                                      self.bias_mu + self.bias_sigma * self.bias_epsilon)
        else:
            return nn.functional.linear(input, self.weight_mu, self.bias_mu)

class EnhancedDuelingDQN(nn.Module):
    def __init__(self, state_dim, n_actions, use_noisy=True):
        super(EnhancedDuelingDQN, self).__init__()
        self.use_noisy = use_noisy
        
        # Shared feature layers with better architecture
        self.feature_layer = nn.Sequential(
            nn.Linear(state_dim, 512),
            nn.LayerNorm(512),  # Layer normalization for stability
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU()
        )
        
        if use_noisy:
            # Noisy value stream
            self.value_stream = nn.Sequential(
                NoisyLinear(256, 128),
                nn.ReLU(),
                NoisyLinear(128, 1)
            )
            
            # Noisy advantage stream
            self.advantage_stream = nn.Sequential(
                NoisyLinear(256, 128),
                nn.ReLU(),
                NoisyLinear(128, n_actions)
            )
        else:
            # Regular value stream
            self.value_stream = nn.Sequential(
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, 1)
            )
            
            # Regular advantage stream
            self.advantage_stream = nn.Sequential(
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, n_actions)
            )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        features = self.feature_layer(x)
        
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        
        # Q(s,a) = V(s) + A(s,a) - mean(A(s,a'))
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
        
        return q_values
    
    def reset_noise(self):
        if self.use_noisy:
            for module in self.modules():
                if isinstance(module, NoisyLinear):
                    module.reset_noise()

# Prioritized Experience Replay Buffer
class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6, beta_start=0.4, beta_end=1.0, epsilon=1e-6):
        self.capacity = capacity
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.epsilon = epsilon
        self.buffer = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.pos = 0
        self.max_priority = 1.0
    
    def push(self, state, action, reward, next_state, done):
        transition = Transition(state, action, reward, next_state, done)
        
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.pos] = transition
        
        self.priorities[self.pos] = self.max_priority
        self.pos = (self.pos + 1) % self.capacity
    
    def sample(self, batch_size, beta):
        if len(self.buffer) == self.capacity:
            priorities = self.priorities
        else:
            priorities = self.priorities[:len(self.buffer)]
        
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()
        
        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)
        samples = [self.buffer[idx] for idx in indices]
        
        # Importance sampling weights
        total = len(self.buffer)
        weights = (total * probabilities[indices]) ** (-beta)
        weights /= weights.max()
        
        batch = Transition(*zip(*samples))
        return batch, indices, weights
    
    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority + self.epsilon
            self.max_priority = max(self.max_priority, priority + self.epsilon)
    
    def __len__(self):
        return len(self.buffer)

# Enhanced Epsilon-greedy policy with improved exploration
class EnhancedEpsilonGreedyPolicy:
    def __init__(self, eps_start, eps_end, eps_decay):
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
    
    def get_epsilon(self, steps_done):
        return self.eps_end + (self.eps_start - self.eps_end) * np.exp(-1. * steps_done / self.eps_decay)
    
    def select_action(self, state, policy_net, n_actions, steps_done, device, use_noisy=True):
        if use_noisy:
            # With noisy networks, we don't need epsilon-greedy
            state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(device)
            with torch.no_grad():
                q_values = policy_net(state_tensor)
            return int(q_values.argmax(dim=1).item())
        else:
            eps = self.get_epsilon(steps_done)
            if np.random.rand() < eps:
                return np.random.randint(n_actions)
            
            state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(device)
            with torch.no_grad():
                q_values = policy_net(state_tensor)
            return int(q_values.argmax(dim=1).item())

# Learning rate scheduler
class DynamicLearningRateScheduler:
    def __init__(self, optimizer, lr_start, lr_end, decay_start_episode, total_episodes):
        self.optimizer = optimizer
        self.lr_start = lr_start
        self.lr_end = lr_end
        self.decay_start_episode = decay_start_episode
        self.total_episodes = total_episodes
    
    def step(self, episode):
        if episode < self.decay_start_episode:
            lr = self.lr_start
        else:
            # Exponential decay after decay_start_episode
            progress = (episode - self.decay_start_episode) / (self.total_episodes - self.decay_start_episode)
            lr = self.lr_end + (self.lr_start - self.lr_end) * np.exp(-5 * progress)
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        return lr

# Enhanced plotting utilities
class EnhancedPlotManager:
    def __init__(self, config):
        self.config = config
        os.makedirs(f"{config.path}/plots", exist_ok=True)
        self.reward_history = []
        self.loss_history = []
        self.lr_history = []
    
    def update_metrics(self, reward, loss=None, learning_rate=None):
        self.reward_history.append(reward)
        if loss is not None:
            self.loss_history.append(loss)
        if learning_rate is not None:
            self.lr_history.append(learning_rate)
    
    def save_comprehensive_plot(self, episode):
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Reward plot
        axes[0,0].plot(self.reward_history, alpha=0.7, linewidth=1, color='blue')
        if len(self.reward_history) >= 50:
            window_size = min(50, len(self.reward_history))
            moving_avg = np.convolve(self.reward_history, np.ones(window_size)/window_size, mode='valid')
            axes[0,0].plot(range(window_size-1, len(self.reward_history)), moving_avg, 
                          linewidth=2, color='red', label=f'MA({window_size})')
        axes[0,0].set_title('Episode Rewards')
        axes[0,0].set_xlabel('Episode')
        axes[0,0].set_ylabel('Total Reward')
        axes[0,0].grid(True, alpha=0.3)
        axes[0,0].legend()
        
        # Loss plot
        if self.loss_history:
            axes[0,1].plot(self.loss_history, color='orange', alpha=0.7)
            if len(self.loss_history) >= 20:
                window_size = min(20, len(self.loss_history))
                loss_ma = np.convolve(self.loss_history, np.ones(window_size)/window_size, mode='valid')
                axes[0,1].plot(range(window_size-1, len(self.loss_history)), loss_ma, 
                              linewidth=2, color='red')
            axes[0,1].set_title('Training Loss')
            axes[0,1].set_xlabel('Update Step')
            axes[0,1].set_ylabel('Loss')
            axes[0,1].grid(True, alpha=0.3)
        
        # Learning rate plot
        if self.lr_history:
            axes[1,0].plot(self.lr_history, color='green')
            axes[1,0].set_title('Learning Rate Schedule')
            axes[1,0].set_xlabel('Episode')
            axes[1,0].set_ylabel('Learning Rate')
            axes[1,0].grid(True, alpha=0.3)
        
        # Performance statistics
        if len(self.reward_history) >= 100:
            recent_rewards = self.reward_history[-100:]
            axes[1,1].hist(recent_rewards, bins=20, alpha=0.7, color='purple')
            axes[1,1].axvline(np.mean(recent_rewards), color='red', linestyle='--', 
                             label=f'Mean: {np.mean(recent_rewards):.2f}')
            axes[1,1].set_title('Recent Reward Distribution (Last 100 Episodes)')
            axes[1,1].set_xlabel('Reward')
            axes[1,1].set_ylabel('Frequency')
            axes[1,1].legend()
        
        plt.tight_layout()
        filename = f"{self.config.path}/plots/comprehensive_plot_ep{episode:04d}.png"
        plt.savefig(filename, dpi=self.config.figure_dpi, bbox_inches='tight')
        plt.close()

# Environment utilities (same as original)
class EnvironmentHandler:
    @staticmethod
    def handle_reset(env):
        reset_out = env.reset()
        if isinstance(reset_out, tuple) and len(reset_out) == 2:
            state, _ = reset_out
        else:
            state = reset_out
        return state
    
    @staticmethod
    def handle_step(env, action):
        step_out = env.step(action)
        if len(step_out) == 5:
            next_state, reward, terminated, truncated, _ = step_out
            done = terminated or truncated
        else:
            next_state, reward, done, _ = step_out
        return next_state, reward, done

# Enhanced D3QN Agent with all improvements
class EnhancedD3QNAgent:
    def __init__(self, config, action_space, state_dim):
        self.config = config
        self.action_space = action_space
        self.steps_done = 0
        self.update_count = 0
        
        os.makedirs(f"{config.path}/checkpoints", exist_ok=True)
        
        # Networks with noisy layers
        self.policy_net = EnhancedDuelingDQN(state_dim, action_space.n_actions, use_noisy=True).to(config.device)
        self.target_net = EnhancedDuelingDQN(state_dim, action_space.n_actions, use_noisy=True).to(config.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        # Optimizer with weight decay
        self.optimizer = optim.Adam(self.policy_net.parameters(), 
                                  lr=config.lr_start, 
                                  weight_decay=config.weight_decay)
        
        # Learning rate scheduler
        self.lr_scheduler = DynamicLearningRateScheduler(
            self.optimizer, config.lr_start, config.lr_end, 
            config.lr_decay_start, config.num_episodes
        )
        
        # Prioritized experience replay
        self.memory = PrioritizedReplayBuffer(
            config.memory_size, config.per_alpha, 
            config.per_beta_start, config.per_beta_end, config.per_epsilon
        )
        
        self.policy = EnhancedEpsilonGreedyPolicy(config.eps_start, config.eps_end, config.eps_decay)
        self.plot_manager = EnhancedPlotManager(config)
        self.env_handler = EnvironmentHandler()
        
        # Performance monitoring
        self.best_avg_reward = -float('inf')
        self.episodes_since_improvement = 0
        self.recent_rewards = deque(maxlen=config.performance_window)
    
    def get_beta(self, episode):
        # Linearly anneal beta from beta_start to beta_end
        progress = episode / self.config.num_episodes
        return self.config.per_beta_start + progress * (self.config.per_beta_end - self.config.per_beta_start)
    
    def select_action(self, state):
        # Reset noise for exploration (but not during gradient computation)
        if not self.policy_net.training:
            self.policy_net.reset_noise()
        return self.policy.select_action(state, self.policy_net, self.action_space.n_actions, 
                                       self.steps_done, self.config.device, use_noisy=True)
    
    def update_network(self, episode):
        if len(self.memory) < self.config.batch_size:
            return None
        
        beta = self.get_beta(episode)
        batch, indices, weights = self.memory.sample(self.config.batch_size, beta)
        
        state_batch = torch.tensor(np.array(batch.state), dtype=torch.float32, device=self.config.device)
        action_batch = torch.tensor(batch.action, dtype=torch.int64, device=self.config.device)
        reward_batch = torch.tensor(batch.reward, dtype=torch.float32, device=self.config.device)
        next_state_batch = torch.tensor(np.array(batch.next_state), dtype=torch.float32, device=self.config.device)
        done_batch = torch.tensor(batch.done, dtype=torch.float32, device=self.config.device)
        weights_batch = torch.tensor(weights, dtype=torch.float32, device=self.config.device)
        
        # Set networks to appropriate modes
        self.policy_net.train()
        self.target_net.eval()
        
        # Current Q-values
        current_q_values = self.policy_net(state_batch).gather(1, action_batch.unsqueeze(1)).squeeze(1)
        
        # Double DQN with target network
        with torch.no_grad():
            # For target network computations, temporarily switch to eval mode
            self.policy_net.eval()
            next_actions = self.policy_net(next_state_batch).argmax(1)
            next_q_values = self.target_net(next_state_batch).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            target_q_values = reward_batch + (self.config.gamma * next_q_values * (1 - done_batch))
            # Switch back to train mode
            self.policy_net.train()
        
        # Compute weighted loss for PER
        td_errors = target_q_values - current_q_values
        loss = (weights_batch * (td_errors ** 2)).mean()
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.config.grad_clip)
        self.optimizer.step()
        
        # Update priorities
        priorities = torch.abs(td_errors).detach().cpu().numpy()
        self.memory.update_priorities(indices, priorities)
        
        self.update_count += 1
        return loss.item()
    
    def soft_update_target_network(self):
        # Soft update: θ_target = τ*θ_local + (1-τ)*θ_target
        for target_param, local_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(self.config.soft_update_tau * local_param.data + 
                                  (1.0 - self.config.soft_update_tau) * target_param.data)
    
    def hard_update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def check_early_stopping(self):
        if len(self.recent_rewards) == self.config.performance_window:
            avg_reward = np.mean(self.recent_rewards)
            if avg_reward > self.best_avg_reward + self.config.min_improvement:
                self.best_avg_reward = avg_reward
                self.episodes_since_improvement = 0
                return False
            else:
                self.episodes_since_improvement += 1
                return self.episodes_since_improvement >= self.config.patience
        return False
    
    def train(self, env):
        episode_rewards = []
        
        for episode in trange(self.config.num_episodes, desc="Episodes"):
            state = self.env_handler.handle_reset(env)
            total_reward = 0.0
            episode_loss = []
            
            # Update learning rate
            current_lr = self.lr_scheduler.step(episode)
            
            for step in range(1, self.config.max_steps + 1):
                # Set policy network to eval mode for action selection
                self.policy_net.eval()
                
                # Select and perform action
                action_idx = self.select_action(state)
                action = self.action_space.actions[action_idx]
                
                # Environment step
                next_state, reward, done = self.env_handler.handle_step(env, action)
                
                # Store transition
                self.memory.push(state, action_idx, reward, next_state, done)
                state = next_state
                total_reward += reward
                self.steps_done += 1
                
                # Train the network (will set to train mode internally)
                loss = self.update_network(episode)
                if loss is not None:
                    episode_loss.append(loss)
                
                # Soft update target network
                if self.update_count % 4 == 0:  # Update every 4 steps
                    self.soft_update_target_network()
                
                if done:
                    break
            
            episode_rewards.append(total_reward)
            self.recent_rewards.append(total_reward)
            
            # Update metrics
            avg_loss = np.mean(episode_loss) if episode_loss else None
            self.plot_manager.update_metrics(total_reward, avg_loss, current_lr)
            
            # Hard update target network periodically
            if episode % self.config.target_update_freq == 0:
                self.hard_update_target_network()
            
            # Save checkpoints and plots
            if (episode + 1) % self.config.plot_freq == 0:
                self.plot_manager.save_comprehensive_plot(episode + 1)
                self.save_checkpoint(episode + 1, episode_rewards)
            
            # # Early stopping check
            if self.check_early_stopping():
                print(f"\nEarly stopping at episode {episode}. No improvement for {self.config.patience} episodes.")
                # break
            
            # Progress reporting
            if len(self.recent_rewards) >= 10:
                recent_avg = np.mean(list(self.recent_rewards)[-10:])
                print(f"Episode {episode:4d} — Reward: {total_reward:7.2f} — Recent Avg: {recent_avg:7.2f} — LR: {current_lr:.2e}")
            else:
                print(f"Episode {episode:4d} — Reward: {total_reward:7.2f} — LR: {current_lr:.2e}")
        
        return episode_rewards
    
    def save_checkpoint(self, episode, episode_rewards):
        checkpoint = {
            'episode': episode,
            'model_state_dict': self.policy_net.state_dict(),
            'target_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'episode_rewards': episode_rewards,
            'best_avg_reward': self.best_avg_reward,
            'config': self.config.__dict__
        }
        
        filename = f"{self.config.path}/checkpoints/checkpoint_ep{episode:04d}.pth"
        torch.save(checkpoint, filename)
        print(f"Checkpoint saved: {filename}")

def main():
    config = Config()
    print(f"Using device: {config.device}")
    
    env = gym.make("BipedalWalker-v3")
    action_space = ActionSpace(config.n_bins)
    state_dim = env.observation_space.shape[0]
    
    print(f"State dimension: {state_dim}")
    print(f"Number of discrete actions: {action_space.n_actions}")
    
    agent = EnhancedD3QNAgent(config, action_space, state_dim)
    episode_rewards = agent.train(env)
    
    # Final comprehensive plot
    agent.plot_manager.save_comprehensive_plot(len(episode_rewards))
    
    env.close()
    print("Training completed!")

# if __name__ == "__main__":
#     main()

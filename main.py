import torch
import gym
import numpy as np
from d3qn import Config, ActionSpace # adjust import if needed
# from ddqn import DQN
from dqn import DQN

def evaluate(render=True, model_path=None):
    # Setup
    config = Config()
    env = gym.make("BipedalWalker-v3", render_mode="human" if render else None)
    action_space = ActionSpace(config.n_bins)
    state_dim = env.observation_space.shape[0]

    # Load model
    policy_net = DQN(state_dim, action_space.n_actions).to(config.device)
    
    # if model_path is None:
    model_path = "figures & dqn2/checkpoints/policy_net_episode_1600.pt"
    
    checkpoint = torch.load(model_path, map_location=config.device)
    policy_net.load_state_dict(torch.load(model_path))
    policy_net.eval()

    # Run 1 episode
    state = env.reset()[0]  # gymnasium returns (obs, info)
    done = False
    total_reward = 0.0

    while not done:
        state_tensor = torch.tensor(state, dtype=torch.float32, device=config.device).unsqueeze(0)
        with torch.no_grad():
            q_values = policy_net(state_tensor)
        action_idx = q_values.argmax(dim=1).item()
        action = action_space.actions[action_idx]

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        state = next_state
        total_reward += reward

    print(f"Evaluation completed — Total Reward: {total_reward:.2f}")
    env.close()

if __name__ == "__main__":
    evaluate()

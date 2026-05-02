import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym


class DQN(nn.Module):
    def __init__(self, num_actions):
        super(DQN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(4, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, num_actions)
        )

    def forward(self, x):
        return self.network(x)


def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    env = gym.make("CartPole-v1")
    num_actions = env.action_space.n

    model = DQN(num_actions).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    rewards = []

    for seed in range(20):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        obs, _ = env.reset(seed=seed)
        env.action_space.seed(seed)

        state = obs.astype(np.float32)
        done = False
        total_reward = 0
        step_count = 0

        while not done and step_count < 500:
            state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(device)

            with torch.no_grad():
                action = model(state_tensor).argmax(dim=1).item()

            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            state = next_obs.astype(np.float32)
            total_reward += reward
            step_count += 1

        rewards.append(total_reward)
        print(f"Seed {seed:02d}: reward = {total_reward}")

    mean_reward = np.mean(rewards)

    print("=" * 50)
    print(f"Mean reward over 20 episodes: {mean_reward:.2f}")
    print("=" * 50)

    if mean_reward >= 480:
        print("Task 1 requirement: PASSED")
    else:
        print("Task 1 requirement: NOT PASSED")

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    args = parser.parse_args()

    evaluate(args)
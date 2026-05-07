import torch
import torch.nn as nn
import numpy as np
import random
import gymnasium as gym
import cv2  
import ale_py
import os
from collections import deque
import argparse

gym.register_envs(ale_py)


class DQN(nn.Module):
    def __init__(self, input_channels, num_actions):
        super(DQN, self).__init__()
        self.network = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )

    def forward(self, x):
        return self.network(x / 255.0)


class AtariPreprocessor:
    def __init__(self, frame_stack=4):
        self.frame_stack = frame_stack
        self.frames = deque(maxlen=frame_stack)

    def preprocess(self, obs):
        if len(obs.shape) == 3 and obs.shape[2] == 3:
            gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        else:
            gray = obs
        resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
        return resized

    def reset(self, obs):
        frame = self.preprocess(obs)
        self.frames = deque(
            [frame for _ in range(self.frame_stack)],
            maxlen=self.frame_stack
        )
        return np.stack(self.frames, axis=0)

    def step(self, obs):
        frame = self.preprocess(obs)
        self.frames.append(frame.copy())
        return np.stack(self.frames, axis=0)


def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    env = gym.make("ALE/Pong-v5", render_mode="rgb_array")
    env.action_space.seed(args.seed)
    env.observation_space.seed(args.seed)

    preprocessor = AtariPreprocessor()
    num_actions = env.action_space.n

    model = DQN(4, num_actions).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    os.makedirs(args.output_dir, exist_ok=True)

    rewards = []

    for ep in range(args.episodes):
        seed = args.seed + ep
        obs, _ = env.reset(seed=seed)
        state = preprocessor.reset(obs)

        done = False
        total_reward = 0
        step_count = 0
        frames = []

        while not done and step_count < args.max_episode_steps:
            if args.save_video and ep < args.video_episodes:
                frame = env.render()
                frames.append(frame)

            state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(device)

            with torch.no_grad():
                action = model(state_tensor).argmax(dim=1).item()

            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            total_reward += reward
            state = preprocessor.step(next_obs)
            step_count += 1

        rewards.append(total_reward)
        print(f"Seed {seed}: reward = {total_reward}")

        
        if args.save_video and ep < args.video_episodes and len(frames) > 0:
            out_path = os.path.join(args.output_dir, f"eval_ep{ep}_reward_{total_reward}.mp4")
            
            
            height, width, _ = frames[0].shape
            
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video = cv2.VideoWriter(out_path, fourcc, 30.0, (width, height))
            
            for f in frames:
                # 顏色轉換
                bgr_frame = cv2.cvtColor(f, cv2.COLOR_RGB2BGR)
                video.write(bgr_frame)
                
            video.release() 
            print(f"Saved video: {out_path}")
        

    mean_reward = np.mean(rewards)
    std_reward = np.std(rewards)

    print("=" * 60)
    print(f"Model path: {args.model_path}")
    print(f"Evaluation episodes: {args.episodes}")
    print(f"Mean reward: {mean_reward:.2f}")
    print(f"Std reward: {std_reward:.2f}")
    print("=" * 60)

    if mean_reward >= 19:
        print("Pong requirement: PASSED")
    else:
        print("Pong requirement: NOT PASSED")

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True, help="Path to trained .pt model")
    parser.add_argument("--output-dir", type=str, default="./eval_videos")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=313551076)
    parser.add_argument("--max-episode-steps", type=int, default=10000)

    # 是否存影片?
    parser.add_argument("--save-video", action="store_true")
    parser.add_argument("--video-episodes", type=int, default=1)

    args = parser.parse_args()
    evaluate(args)
#The ManualPreprocessWrapper is a custom Gymnasium 
# environment wrapper designed to prepare raw visual 
# observations from Atari games for deep reinforcement 
# learning. It performs three key preprocessing steps: 
# converting RGB frames to grayscale, resizing them to 
# 84×84 pixels, and stacking the most recent frames to
# form a fixed-size input. This ensures that the agent 
# receives a compact yet informative state representation 
# that captures both spatial and temporal information,
# enabling more effective learning in environments like 
# Atari Bowling.


import gymnasium as gym
import numpy as np
import cv2
import torchvision.transforms as T
import torch

from collections import deque


class ManualPreprocessWrapper(gym.Wrapper):
    """
    Wrapper to grayscale, resize, and stack frames for visual input (now GPU-accelerated).
    """
    def __init__(self, env, frame_stack=4, device='cuda'):
        super().__init__(env)
        self.frame_stack = frame_stack
        self.device = torch.device(device)
        self.frames = deque(maxlen=frame_stack)

        self.transform = T.Compose([
            T.ToPILImage(),
            T.Grayscale(),
            T.Resize((84, 84), interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),  # output: (1, 84, 84) float32 [0,1]
        ])

        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(frame_stack, 84, 84),
            dtype=np.uint8
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        processed = self._preprocess(obs)
        for _ in range(self.frame_stack):
            self.frames.append(processed)
        return np.stack(self.frames, axis=0), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        processed = self._preprocess(obs)
        self.frames.append(processed)
        return np.stack(self.frames, axis=0), reward, terminated, truncated, info

    def _preprocess(self, obs):
        # Transform on GPU, return as uint8 NumPy array for compatibility
        tensor = self.transform(obs).to(self.device)  # shape: (1, 84, 84)
        return (tensor * 255).byte().squeeze(0).cpu().numpy()  # shape: (84, 84)

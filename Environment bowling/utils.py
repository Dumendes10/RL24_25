import gymnasium as gym
import numpy as np
import cv2
import torchvision.transforms as T
import torch

from collections import deque

class ShapedRewardWrapper(gym.Wrapper):
    """
    Wrapper to apply reward shaping based on frame differences,
    providing additional signal when visual change (e.g., pins falling) occurs.
    """
    def __init__(self, env):
        super().__init__(env)
        self.last_frame = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.last_frame = obs
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Reward shaping: bonus for any visual change
        shaped_reward = reward
        if self.last_frame is not None:
            diff = np.mean(np.abs(obs.astype(np.float32) - self.last_frame.astype(np.float32)))
            shaped_bonus = diff / 255.0  # Normalize to [0,1]
            shaped_reward += shaped_bonus

        self.last_frame = obs
        return obs, shaped_reward, terminated, truncated, info


#This cell defines a Replay Buffer, which is a key component in Deep Q-Learning (and Rainbow DQN). It stores the agent’s experiences and allows the model to train on a random sample of past transitions, breaking the correlation between consecutive frames and improving learning stability.
# from collections import deque
# import random
# import torch

# class ReplayBuffer:
#     def __init__(self, capacity=100_000):
#         self.buffer = deque(maxlen=capacity)

#     def add(self, s, a, r, s2, d):
#         self.buffer.append((s, a, r, s2, d))

#     def sample(self, batch_size):
#         batch = random.sample(self.buffer, batch_size)
#         states, actions, rewards, next_states, dones = zip(*batch)
#         return (
#             torch.tensor(states, dtype=torch.float32),
#             torch.tensor(actions),
#             torch.tensor(rewards, dtype=torch.float32),
#             torch.tensor(next_states, dtype=torch.float32),
#             torch.tensor(dones, dtype=torch.float32)
#         )

#     def __len__(self):
#         return len(self.buffer)




# In reinforcement learning, experience replay allows the agent 
# to store past experiences (state, action, reward, next state, 
# done) and sample them later for training. Prioritized replay 
# improves upon this by sampling more important experiences more 
# often, based on their learning signal (typically the TD error).
# In reinforcement learning, experience replay allows the agent to 
# store past experiences (state, action, reward, next state, done) 
# and sample them later for training. Prioritized replay improves
# upon this by sampling more important experiences more often, based 
# on their learning signal (typically the TD error).

import numpy as np
from collections import deque, namedtuple
import random
import torch

class SumTree:
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.empty(capacity, dtype=object)
        self.index = 0
        self.n_entries = 0

    def add(self, priority, data):
        tree_idx = self.index + self.capacity - 1
        self.data[self.index] = data
        self.update(tree_idx, priority)

        self.index = (self.index + 1) % self.capacity
        self.n_entries = min(self.n_entries + 1, self.capacity)

    def update(self, tree_idx, priority):
        change = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority
        while tree_idx != 0:
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += change

    def get(self, value):
        idx = 0
        while idx < self.capacity - 1:
            left = 2 * idx + 1
            right = left + 1
            idx = left if value <= self.tree[left] else right
            value -= self.tree[left] if value > self.tree[left] else 0
        data_idx = idx - self.capacity + 1
        return idx, self.tree[idx], self.data[data_idx]

    def total(self):
        return self.tree[0]

class PrioritizedReplayBuffer:
    def __init__(self, capacity=100000, alpha=0.6, beta=0.4, beta_increment=1e-6, epsilon=1e-5):
        self.tree = SumTree(capacity)
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.epsilon = epsilon
        self.max_priority = 1.0

    def add(self, s, a, r, s2, d):
        transition = (s, a, r, s2, d)
        self.tree.add(self.max_priority ** self.alpha, transition)

    def sample(self, batch_size):
        batch = []
        idxs = []
        priorities = []
        segment = self.tree.total() / batch_size
        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            value = random.uniform(a, b)
            idx, p, data = self.tree.get(value)
            batch.append(data)
            idxs.append(idx)
            priorities.append(p)

        sampling_probabilities = np.array(priorities) / self.tree.total()
        weights = (self.tree.n_entries * sampling_probabilities) ** (-self.beta)
        self.beta = min(1.0, self.beta + self.beta_increment)
        weights = torch.tensor(weights / weights.max(), dtype=torch.float32)

        s, a, r, s2, d = zip(*batch)
        return (
            torch.tensor(s, dtype=torch.float32),
            torch.tensor(a),
            torch.tensor(r, dtype=torch.float32),
            torch.tensor(s2, dtype=torch.float32),
            torch.tensor(d, dtype=torch.float32),
            weights,
            idxs
        )

    def update_priorities(self, idxs, errors):
        for idx, error in zip(idxs, errors.detach().cpu().numpy()):
            priority = (abs(error) + self.epsilon) ** self.alpha
            self.tree.update(idx, priority)

    def __len__(self):
        return self.tree.n_entries



# Add N step support (n=3)
n_step = 3
gamma = 0.99
n_gamma = gamma ** n_step
n_step_buffer = deque(maxlen=n_step)

def compute_n_step_transition(buffer):
    s, a = buffer[0][:2]
    r, s2, d = 0, None, False
    for i, (_, _, ri, si, di) in enumerate(buffer):
        r += (gamma ** i) * ri
        s2, d = si, di
        if d: break
    return s, a, r, s2, d


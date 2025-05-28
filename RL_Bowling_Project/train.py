
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from collections import deque, namedtuple
import random
from agents.rainbow_dqn_agent import RainbowDQN
from environment.bowling_env import make_bowling_env

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, *args):
        self.buffer.append(Transition(*args))

    def sample(self, batch_size):
        transitions = random.sample(self.buffer, batch_size)
        return Transition(*zip(*transitions))

    def __len__(self):
        return len(self.buffer)

def preprocess(obs):
    # Exemplo básico (podes adaptar com grayscale, resize etc.)
    obs = obs.transpose((2, 0, 1))  # HWC -> CHW
    obs = torch.tensor(obs, dtype=torch.float32) / 255.0
    return obs.unsqueeze(0)  # Add batch dimension

def train(num_episodes=500, buffer_size=10000, batch_size=32, gamma=0.99, lr=1e-4, epsilon_start=1.0, epsilon_final=0.1, epsilon_decay=500):
    env = make_bowling_env()
    obs, _ = env.reset()
    input_shape = preprocess(obs).shape[1:]
    n_actions = env.action_space.n

    policy_net = RainbowDQN(input_shape, n_actions)
    optimizer = optim.Adam(policy_net.parameters(), lr=lr)
    replay_buffer = ReplayBuffer(buffer_size)

    epsilon = epsilon_start
    episode_rewards = []

    for episode in range(num_episodes):
        obs, _ = env.reset()
        state = preprocess(obs)
        total_reward = 0

        done = False
        while not done:
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    q_values = policy_net(state)
                    action = q_values.argmax().item()

            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            next_state = preprocess(next_obs)

            replay_buffer.push(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

            if len(replay_buffer) >= batch_size:
                transitions = replay_buffer.sample(batch_size)
                batch = Transition(*transitions)

                non_final_mask = torch.tensor(tuple(map(lambda d: not d, batch.done)), dtype=torch.bool)
                non_final_next_states = torch.cat([s for s, d in zip(batch.next_state, batch.done) if not d])

                state_batch = torch.cat(batch.state)
                action_batch = torch.tensor(batch.action).unsqueeze(1)
                reward_batch = torch.tensor(batch.reward, dtype=torch.float32)
                done_batch = torch.tensor(batch.done, dtype=torch.float32)

                q_values = policy_net(state_batch).gather(1, action_batch)

                next_q_values = torch.zeros(batch_size)
                if non_final_next_states.size(0) > 0:
                    next_q_values[non_final_mask] = policy_net(non_final_next_states).max(1)[0].detach()

                expected_q_values = reward_batch + gamma * next_q_values * (1 - done_batch)
                loss = F.mse_loss(q_values.squeeze(), expected_q_values)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        epsilon = max(epsilon_final, epsilon_start - episode / epsilon_decay)
        episode_rewards.append(total_reward)
        print(f"Episode {episode + 1}, Reward: {total_reward}, Epsilon: {epsilon:.3f}")

    env.close()
    return episode_rewards

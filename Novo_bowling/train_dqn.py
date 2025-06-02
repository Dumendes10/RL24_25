import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import matplotlib.pyplot as plt
import cv2
from collections import deque
from agents import *
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def preprocess(obs):
    gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, (84, 84))
    norm = resized / 255.0
    return np.expand_dims(norm, axis=0)  # (1, 84, 84)

def moving_average(data, window=10):
    return np.convolve(data, np.ones(window) / window, mode="valid")

def train_dqn(episodes=500, batch_size=32):
    env = gym.make("ALE/Bowling-v5", render_mode="rgb_array")
    obs, _ = env.reset()
    state = preprocess(obs)
    state_shape = state.shape
    n_actions = env.action_space.n

    policy_net = DQN(state_shape, n_actions).to(device)
    target_net = DQN(state_shape, n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=1e-4)
    buffer = ReplayBuffer(10000)

    gamma = 0.99
    epsilon = 1.0
    epsilon_min = 0.05
    epsilon_decay = 0.995
    sync_target_steps = 1000
    total_steps = 0
    rewards = []

    for ep in range(episodes):
        obs, _ = env.reset()
        state = preprocess(obs)
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        total_reward = 0
        done = False

        while not done:
            total_steps += 1
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    q_values = policy_net(state)
                    action = q_values.max(1)[1].item()

            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            next_state_np = preprocess(next_obs)
            next_state = torch.FloatTensor(next_state_np).unsqueeze(0).to(device)

            buffer.push((state, action, reward, next_state, done))
            state = next_state
            total_reward += reward

            # Treino
            if len(buffer) >= batch_size:
                batch = buffer.sample(batch_size)
                b_states, b_actions, b_rewards, b_next_states, b_dones = zip(*batch)

                b_states = torch.cat(b_states)
                b_actions = torch.LongTensor(b_actions).unsqueeze(1).to(device)
                b_rewards = torch.FloatTensor(b_rewards).to(device)
                b_next_states = torch.cat(b_next_states)
                b_dones = torch.BoolTensor(b_dones).to(device)

                q_values = policy_net(b_states).gather(1, b_actions).squeeze(1)
                next_q = target_net(b_next_states).max(1)[0]
                next_q[b_dones] = 0.0
                expected_q = b_rewards + gamma * next_q

                loss = nn.MSELoss()(q_values, expected_q.detach())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if total_steps % sync_target_steps == 0:
                target_net.load_state_dict(policy_net.state_dict())

        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        rewards.append(total_reward)
        print(f"Episode {ep+1} - Total reward: {total_reward:.2f} - Epsilon: {epsilon:.3f}")

    os.makedirs("Graphics", exist_ok=True)

    # Gráfico
    plt.figure(figsize=(10, 5))
    plt.plot(rewards, alpha=0.4, label="Recompensa por Episódio")
    if len(rewards) >= 10:
        plt.plot(moving_average(rewards), label="Média Móvel", linewidth=2)
    plt.xlabel("Episódio")
    plt.ylabel("Recompensa Total")
    plt.title("DQN - Recompensa por Episódio (Bowling)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("Graphics/reward_plot_DQN.png")
    plt.show()

#from bowling_env import make_bowling_env
import ale_py 
from agents import QLearningAgent
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import os

def moving_average(data, window=10):
    return np.convolve(data, np.ones(window) / window, mode="valid")

def train_q_learning(episodes=500):
    env = gym.make("ALE/Bowling-v5", render_mode="rgb_array")
    obs_space = 1000  # simplificação via hash discretizado
    action_space = env.action_space.n
    agent = QLearningAgent(obs_space, action_space)

    rewards = []

    for ep in range(episodes):
        obs, _ = env.reset()
        state = int(np.sum(obs)) % obs_space

        total_reward = 0
        done = False

        while not done:
            action = agent.choose_action(state)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            next_state = int(np.sum(next_obs)) % obs_space  # corrigido: era obs, agora next_obs

            agent.update(state, action, reward, next_state)
            state = next_state
            total_reward += reward

        agent.decay_epsilon()
        rewards.append(total_reward)
        print(f"Episode {ep+1}: Total reward = {total_reward:.2f}")

    # Criar diretório se não existir
    os.makedirs("Graphics", exist_ok=True)

    # Gráfico da recompensa por episódio
    plt.figure(figsize=(10, 5))
    plt.plot(rewards, alpha=0.4, label="Recompensa por Episódio")
    if len(rewards) >= 10:
        plt.plot(moving_average(rewards), label="Média Móvel", linewidth=2)
    plt.xlabel("Episódio")
    plt.ylabel("Recompensa Total")
    plt.title("Evolução da Recompensa - Q-Learning no Bowling")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("Graphics/reward_plot_Q_Learning.png")
    plt.show()

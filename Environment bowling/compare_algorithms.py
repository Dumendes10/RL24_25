import numpy as np
import matplotlib.pyplot as plt
import os

def comparar_algoritmos():
    # Criar diretórios se não existirem
    os.makedirs("results", exist_ok=True)
    os.makedirs("Graphics", exist_ok=True)

    # Carregar recompensas salvas
    rewards_q = np.load("results/rewards_q_learning.npy")
    rewards_dqn = np.load("results/rewards_dqn.npy")
    rewards_ppo = np.load("results/rewards_ppo.npy")
    # rewards_a2c = np.load("results/rewards_a2c.npy")

    # Função de média móvel
    def moving_average(data, window=10):
        return np.convolve(data, np.ones(window)/window, mode="valid")

    # Gráfico de comparação
    plt.figure(figsize=(12, 6))

    plt.plot(rewards_q, alpha=0.3, label="Q-Learning", color="#1f77b4")
    plt.plot(rewards_dqn, alpha=0.3, label="DQN", color="#d62728")
    plt.plot(rewards_ppo, alpha=0.3, label="PPO", color="#2ca02c")
# plt.plot(rewards_a2c, alpha=0.3, label="A2C", color="#d62728")

    #plt.plot(moving_average(rewards_q), label="Q-Learning (Média)", linewidth=2)
    #plt.plot(moving_average(rewards_dqn), label="DQN (Média)", linewidth=2)
    #plt.plot(moving_average(rewards_ppo), label="PPO (Média)", linewidth=2)
    # plt.plot(moving_average(rewards_a2c), label="A2C (Média)", linewidth=2)

    plt.xlabel("Episódio")
    plt.ylabel("Recompensa Total")
    plt.title("Comparação de Algoritmos - Recompensa por Episódio")
    plt.legend()
   # plt.grid(True)
    plt.tight_layout()
    plt.savefig("Graphics/compare_algorithms_plot.png")
    plt.close()

    # Histograma final
    plt.figure(figsize=(10, 5))
    plt.hist(rewards_q[-50:], alpha=0.5, label="Q-Learning", color="#1f77b4")
    plt.hist(rewards_dqn[-50:], alpha=0.5, label="DQN", color="#d62728")
    plt.hist(rewards_ppo[-50:], alpha=0.5, label="PPO", color="#2ca02c")
    # plt.hist(rewards_a2c[-50:], alpha=0.5, label="A2C", color='red')

    plt.xlabel("Recompensa")
    plt.ylabel("Frequência")
    plt.title("Distribuição de Recompensas (últimos 50 episódios)")
    plt.legend()
    # plt.grid(True)
    plt.tight_layout()
    plt.savefig("Graphics/compare_histograms.png")
    plt.close()
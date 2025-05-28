
import matplotlib.pyplot as plt
import numpy as np

def moving_average(data, window):
    return np.convolve(data, np.ones(window)/window, mode='valid')

def plot_rewards(episode_rewards, title="Reward per Episode", window=10):
    plt.figure(figsize=(10, 5))
    plt.plot(episode_rewards, label="Episode Reward", color='steelblue')
    if len(episode_rewards) >= window:
        ma = moving_average(episode_rewards, window)
        plt.plot(range(window - 1, len(episode_rewards)), ma,
                 label=f"Moving Avg ({window})", linestyle="--", color='orange')
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_convergence(rewards, threshold, window=10):
    plt.figure(figsize=(10, 5))
    plt.plot(rewards, label="Episode Reward", color='steelblue')
    plt.axhline(y=threshold, color='red', linestyle=':', label=f"Threshold = {threshold}")
    if len(rewards) >= window:
        ma = moving_average(rewards, window)
        plt.plot(range(window - 1, len(rewards)), ma,
                 label=f"Moving Avg ({window})", linestyle="--", color='orange')
        for i in range(len(ma)):
            if all(m >= threshold for m in ma[i:i+window]):
                plt.axvline(x=i + window, color='green', linestyle='--', label="Convergence Point")
                break
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Convergence Analysis")
    plt.legend()
    plt.grid(True)
    plt.show()

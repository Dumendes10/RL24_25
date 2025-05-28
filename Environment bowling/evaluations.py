
import torch
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Global reward tracking list
episode_rewards = []

# Shared: Moving average utility
def moving_average(data, window=10):
    return np.convolve(data, np.ones(window) / window, mode="valid")

# Track reward and episode state during training
def track_reward(episode_reward, episode_count, reward, done, step, env, next_state):
    episode_reward += reward
    if done:
        episode_rewards.append(episode_reward)
        episode_count += 1
        print(f" Episode {episode_count} finished at step {step} — Reward: {episode_reward:.2f}")
        episode_reward = 0
        state, _ = env.reset()
    else:
        state = next_state
    return episode_reward, episode_count, state

# Save rewards to CSV
def save_rewards_csv(filename="rewards.csv"):
    df = pd.DataFrame({
        "episode": range(1, len(episode_rewards) + 1),
        "reward": episode_rewards
    })
    df.to_csv(filename, index=False)
    print(f" Saved reward log to {filename}")

# Plot reward curve with moving average
def plot_rewards(title="Reward per Episode", window=10):
    plt.figure(figsize=(10, 5))
    plt.plot(episode_rewards, label="Episode Reward", color='steelblue')
    if len(episode_rewards) >= window:
        ma = moving_average(episode_rewards, window)
        plt.plot(range(window - 1, len(episode_rewards)), ma, label=f"Moving Avg ({window})", linestyle="--", color='orange')
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

# Evaluate agent's performance after training (pure exploitation)
def evaluate_agent(agent, env, episodes=5, device="cpu"):
    total_rewards = []
    for ep in range(episodes):
        state, _ = env.reset()
        done = False
        ep_reward = 0
        while not done:
            with torch.no_grad():
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
                action = agent.act(state_tensor)
            state, reward, terminated, truncated, _ = env.step(action)
            ep_reward += reward
            done = terminated or truncated
        total_rewards.append(ep_reward)
        print(f" Evaluation Episode {ep + 1}: Reward = {ep_reward:.2f}")
    avg = np.mean(total_rewards)
    print(f"\n Average Reward over {episodes} episodes: {avg:.2f}")
    return total_rewards

# Check for convergence using moving average
def estimate_convergence_verbose(rewards, threshold=5.0, window=10):
    print(f"\n Checking convergence: threshold = {threshold}, window = {window} episodes")

    if len(rewards) < window:
        print(" Not enough episodes to compute moving average.")
        return None

    ma = moving_average(rewards, window)
    converged = False

    for i, avg in enumerate(ma):
        if avg >= threshold:
            print(f" Convergence detected at episode {i + window}:")
            print(f"   • Moving Average: {avg:.2f}")
            print(f"   • Range: Episodes {i + 1} to {i + window}")
            converged = True
            break

    if not converged:
        print(" Agent did not reach convergence threshold during training.")

    #  Plot
    plt.figure(figsize=(10, 5))
    plt.plot(rewards, label="Episode Reward", color='steelblue')
    plt.axhline(y=threshold, color='red', linestyle=':', label=f"Threshold = {threshold}")
    if len(rewards) >= window:
        plt.plot(range(window - 1, len(rewards)), ma, label=f"Moving Avg ({window})", linestyle="--", color='orange')
        if converged:
            plt.axvline(x=i + window, color='green', linestyle='--', label="Convergence Point")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Convergence Analysis")
    plt.legend()
    plt.grid(True)
    plt.show()

    return i + window if converged else None

#  Compute success rate based on reward threshold
def compute_success_rate(rewards, threshold=5.0):
    successes = [r for r in rewards if r >= threshold]
    rate = len(successes) / len(rewards) if rewards else 0.0
    print(f" Success Rate: {rate * 100:.2f}% ({len(successes)} / {len(rewards)} episodes ≥ {threshold} points)")
    return rate

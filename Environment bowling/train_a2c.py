import gymnasium as gym
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage, VecFrameStack
from stable_baselines3.common.monitor import Monitor
import matplotlib.pyplot as plt
import os
import numpy as np

def train_a2c(episodes=20):
    # Criar ambiente com monitor e wrappers
    env = gym.make("ALE/Bowling-v5", render_mode="rgb_array")
    env = Monitor(env)
    env = DummyVecEnv([lambda: env])
    env = VecTransposeImage(env)
    env = VecFrameStack(env, n_stack=4)

    model = A2C(
        "CnnPolicy",
        env,
        verbose=1,
        n_steps=2048  ,  # rollout mais curto para debug rápido
        learning_rate=7e-4,
        tensorboard_log="./a2c_logs/"
    )

    total_timesteps = 500000
    model.learn(total_timesteps=total_timesteps)

    # Avaliação rápida
    rewards = []
    eval_env = gym.make("ALE/Bowling-v5", render_mode="rgb_array")
    eval_env = DummyVecEnv([lambda: eval_env])
    eval_env = VecTransposeImage(eval_env)
    eval_env = VecFrameStack(eval_env, n_stack=4)

    for ep in range(episodes):
        obs = eval_env.reset()
        done = False
        total_reward = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = eval_env.step(action)
            done = done[0]               # extrair valor escalar do array
            reward = reward[0]           # idem
            total_reward += reward
        rewards.append(total_reward)
        print(f"Episode {ep+1}: Total reward = {total_reward:.2f}")

    
    os.makedirs("Graphics", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    np.save("results/rewards_a2c.npy", rewards)

    def moving_average(data, window=10):
        return np.convolve(data, np.ones(window)/window, mode="valid")

    plt.figure(figsize=(10, 5))
    plt.plot(rewards, alpha=0.4, label="Recompensa por Episódio")
    if len(rewards) >= 2:
        ma = moving_average(rewards, window=min(5, len(rewards)))
        plt.plot(ma, label="Média Móvel", linewidth=2)
    plt.xlabel("Episódio")
    plt.ylabel("Recompensa Total")
    plt.title("A2C - Recompensa por Episódio (versão leve)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("Graphics/reward_plot_A2C_debug.png")
  #  plt.show()
    plt.close()         



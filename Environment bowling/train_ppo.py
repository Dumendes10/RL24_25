import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage, VecFrameStack
from stable_baselines3.common.monitor import Monitor
import matplotlib.pyplot as plt
import os
import numpy as np

def train_ppo(episodes=150):
    # Criar ambiente com monitor e wrappers
    env = gym.make("ALE/Bowling-v5", render_mode="rgb_array")
    env = Monitor(env)
    env = DummyVecEnv([lambda: env])
    env = VecTransposeImage(env)
    env = VecFrameStack(env, n_stack=4)  # empilhar 4 frames

    model = PPO(
        "CnnPolicy",
        env,
        verbose=1,
        n_steps= 1024,  # mais passos por rollout
        batch_size= 64,
        learning_rate=2.5e-4,
        tensorboard_log="./ppo_logs/"
    )

    total_timesteps =  500000 # mais tempo de treino
    model.learn(total_timesteps=total_timesteps)

    # Avaliação
    rewards = []
    eval_env = gym.make("ALE/Bowling-v5", render_mode="rgb_array")
    eval_env = DummyVecEnv([lambda: eval_env])
    eval_env = VecTransposeImage(eval_env)
    eval_env = VecFrameStack(eval_env, n_stack=4)
    
    for ep in range(150):
        obs = eval_env.reset()
        done = False
        total_reward = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = eval_env.step(action)
            done = done[0]
            reward = reward[0]
            total_reward += reward
        rewards.append(total_reward)
        print(f"Episode {ep+1}: Total reward = {total_reward:.2f}")
        
    
    os.makedirs("Graphics", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    np.save("results/rewards_ppo.npy", rewards)

    # Gráfico
    def moving_average(data, window=10):
        return np.convolve(data, np.ones(window)/window, mode="valid")

    plt.figure(figsize=(10, 5))
    plt.plot(rewards, alpha=0.4, label="Recompensa por Episódio")
    if len(rewards) >= 10:
        plt.plot(moving_average(rewards), label="Média Móvel", linewidth=2)
    plt.xlabel("Episódio")
    plt.ylabel("Recompensa Total")
    plt.title("PPO - Recompensa por Episódio (Bowling)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("Graphics/reward_plot_PPO.png")
#    plt.show()
    plt.close()         

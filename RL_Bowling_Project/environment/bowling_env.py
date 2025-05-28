import gymnasium as gym
import ale_py 

def make_bowling_env():
    gym.register_envs(ale_py)  # required for Gymnasium to recognize ALE/Bowling-v5

    env = gym.make("ALE/Bowling-v5", render_mode="rgb_array")
    return env

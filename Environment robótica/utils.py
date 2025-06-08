import numpy as np
import torch
import os

def evaluate_policy(policy, env, eval_episodes=10, render=False):
    """
    Evaluate the policy in the given environment
    
    Args:
        policy: The policy to evaluate
        env: The environment to evaluate in
        eval_episodes: Number of episodes to evaluate
        render: Whether to render the environment
        
    Returns:
        avg_reward: Average reward over eval_episodes
        success_rate: Success rate over eval_episodes
    """
    avg_reward = 0.
    successes = 0
    
    for _ in range(eval_episodes):
        obs_dict, _ = env.reset()
        obs = obs_dict['observation']
        done = False
        truncated = False
        
        while not (done or truncated):
            if render:
                env.render()
                
            # Select action - handle both SAC and TD3
            if hasattr(policy, '__class__') and policy.__class__.__name__ == 'SAC':
                action = policy.select_action(np.array(obs), evaluate=True)
            else:
                action = policy.select_action(np.array(obs), noise=0)
            
            obs_dict, reward, done, truncated, info = env.step(action)
            obs = obs_dict['observation']
            avg_reward += reward
            
            if 'is_success' in info and info['is_success'] == 1.0:
                successes += 1
                break
    
    avg_reward /= eval_episodes
    success_rate = successes / eval_episodes
    
    print("---------------------------------------")
    print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}, Success rate: {success_rate:.3f}")
    print("---------------------------------------")
    
    return avg_reward, success_rate

def save_training_data(results, filename):
    """Save training data to file"""
    np.save(filename, results)

def create_directory(directory_path):
    """Create a directory if it doesn't exist"""
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

def print_env_info(env):
    """Print information about the environment"""
    print("---------------------------------------")
    print(f"Environment: {env.unwrapped.spec.id}")
    print(f"Observation Space: {env.observation_space}")
    print(f"Action Space: {env.action_space}")
    print(f"Action Space High: {env.action_space.high}")
    print(f"Action Space Low: {env.action_space.low}")
    print("---------------------------------------")
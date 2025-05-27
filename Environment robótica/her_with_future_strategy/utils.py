import numpy as np
import matplotlib.pyplot as plt
import torch
import time
import os
from IPython.display import clear_output

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
        # Extract the observation from the dictionary
        obs = obs_dict['observation']
        done = False
        truncated = False
        
        while not (done or truncated):
            if render:
                env.render()
                
            # Select action without noise for evaluation
            # Check if the policy is SAC or TD3 and call select_action accordingly
            if hasattr(policy, '__class__') and policy.__class__.__name__ == 'SAC':
                action = policy.select_action(np.array(obs), evaluate=True)
            else:  # Assume TD3 or similar with 'noise' parameter
                action = policy.select_action(np.array(obs), noise=0)
            
            # Perform action
            obs_dict, reward, done, truncated, info = env.step(action)
            # Extract observation from dictionary
            obs = obs_dict['observation']
            avg_reward += reward
            
            if 'is_success' in info:
                if info['is_success'] == 1.0:
                    successes += 1
                    break
    
    avg_reward /= eval_episodes
    success_rate = successes / eval_episodes
    
    print("---------------------------------------")
    print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}, Success rate: {success_rate:.3f}")
    print("---------------------------------------")
    
    return avg_reward, success_rate

def plot_learning_curves(results, labels, smoothing_window=10, title="Learning Curves", show=True, save_path=None):
    """
    Plot learning curves for one or more agents
    
    Args:
        results: List of dictionaries containing training data
        labels: List of labels for each agent
        smoothing_window: Window size for smoothing the curves
        title: Title of the plot
        show: Whether to show the plot
        save_path: Path to save the plot
    """
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
    
    for i, result in enumerate(results):
        # Extract data
        episodes = result.get('episodes', np.arange(len(result.get('rewards', []))))
        rewards = result.get('rewards', [])
        success_rates = result.get('success_rates', [])
        steps = result.get('steps', np.arange(len(rewards)))
        
        # Smooth rewards
        if smoothing_window > 1 and len(rewards) >= smoothing_window:
            weights = np.ones(smoothing_window) / smoothing_window
            rewards_smooth = np.convolve(rewards, weights, mode='valid')
            success_rates_smooth = np.convolve(success_rates, weights, mode='valid') if len(success_rates) >= smoothing_window else success_rates
            episodes_smooth = episodes[smoothing_window-1:] if len(episodes) >= smoothing_window else episodes
        else:
            rewards_smooth = rewards
            success_rates_smooth = success_rates
            episodes_smooth = episodes
        
        # Plot rewards
        ax1.plot(episodes_smooth, rewards_smooth, label=labels[i])
        ax1.set_xlabel('Episodes')
        ax1.set_ylabel('Average Reward')
        ax1.set_title('Reward vs Episodes')
        ax1.legend(loc='lower right')
        ax1.grid(True)
        
        # Plot success rates
        if len(success_rates) > 0:
            ax2.plot(range(len(success_rates_smooth)), success_rates_smooth, label=labels[i])
            ax2.set_xlabel('Evaluation')
            ax2.set_ylabel('Success Rate')
            ax2.set_title('Success Rate vs Evaluation')
            ax2.legend(loc='lower right')
            ax2.grid(True)
        
        # Plot steps
        if len(steps) > 0:
            ax3.plot(episodes, steps, label=labels[i])
            ax3.set_xlabel('Episodes')
            ax3.set_ylabel('Training Steps')
            ax3.set_title('Training Steps vs Episodes')
            ax3.legend(loc='lower right')
            ax3.grid(True)
    
    fig.suptitle(title, fontsize=16)
    plt.tight_layout()
    
    # Save the plot if specified
    if save_path is not None:
        plt.savefig(save_path)
    
    # Show the plot if specified
    if show:
        plt.show()
    else:
        plt.close()

def save_training_data(results, filename):
    """
    Save training data to file
    
    Args:
        results: Dictionary containing training data
        filename: Filename to save to
    """
    np.save(filename, results)

def load_training_data(filename):
    """
    Load training data from file
    
    Args:
        filename: Filename to load from
        
    Returns:
        results: Dictionary containing training data
    """
    return np.load(filename, allow_pickle=True).item()

def create_directory(directory_path):
    """
    Create a directory if it doesn't exist
    
    Args:
        directory_path: Path to the directory
    """
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

def print_env_info(env):
    """
    Print information about the environment
    
    Args:
        env: The environment to get information about
    """
    print("---------------------------------------")
    print(f"Environment: {env.unwrapped.spec.id}")
    print(f"Observation Space: {env.observation_space}")
    print(f"Action Space: {env.action_space}")
    print(f"Action Space High: {env.action_space.high}")
    print(f"Action Space Low: {env.action_space.low}")
    print("---------------------------------------")
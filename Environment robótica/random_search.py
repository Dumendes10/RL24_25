import numpy as np
import gymnasium as gym
import panda_gym
import torch
import os
import matplotlib.pyplot as plt
import json
from datetime import datetime
from utils import evaluate_policy, create_directory, print_env_info, save_training_data
from sac import SAC
from td3 import TD3
from her import HERReplayBuffer
import warnings

warnings.filterwarnings('ignore')

# Define environment
ENV_NAME = "PandaReach-v3"

def train_sac_with_params(params, env_name=ENV_NAME, max_timesteps=30000, eval_freq=5000, 
                          save_models=True, save_dir="./results/sac_random_search", device='cpu',
                          use_her=False, her_strategy="final", her_k=4):
    """
    Train an SAC agent with the given hyperparameters
    
    Args:
        params: Dictionary of hyperparameters
        env_name: Name of the environment
        max_timesteps: Maximum number of timesteps to train for
        eval_freq: Frequency of evaluation
        save_models: Whether to save models
        save_dir: Directory to save models
        device: Device to use for training
        use_her: Whether to use Hindsight Experience Replay
        her_strategy: HER goal selection strategy ('final', 'future', 'random')
        her_k: Number of goals to sample per transition for HER
        
    Returns:
        result: Dictionary containing training data and performance metrics
    """
    # Create save directory
    if save_models and not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Create environment
    env = gym.make(env_name)
    eval_env = gym.make(env_name)
    
    # Set seeds for reproducibility
    seed = 0
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Get state and action dimensions from environment
    state_dim = env.observation_space['observation'].shape[0]
    action_dim = env.action_space.shape[0]
    
    # Initialize policy with hyperparameters
    policy = SAC(
        state_dim=state_dim,
        action_dim=action_dim,
        action_space=env.action_space,
        device=device,
        hidden_dim=params['hidden_dim'],
        lr=params['lr'],
        gamma=params['gamma'],
        tau=params['tau'],
        alpha=params['alpha'],
        automatic_entropy_tuning=params['automatic_entropy_tuning']
    )
    
    # Initialize replay buffer - standard or HER
    if use_her:
        replay_buffer = HERReplayBuffer(
            state_dim=state_dim,
            action_dim=action_dim,
            max_size=params['buffer_size'],
            device=device
        )
    else:
        from replay_buffer import ReplayBuffer
        replay_buffer = ReplayBuffer(
            state_dim=state_dim,
            action_dim=action_dim,
            max_size=params['buffer_size'],
            device=device
        )
    
    # Initialize variables
    episode_reward = 0
    episode_timesteps = 0
    episode_num = 0
    state_dict, _ = env.reset(seed=seed)
    state = state_dict['observation']
    
    # Initialize lists for storing data
    evaluations = []
    success_rates = []
    training_rewards = []
    episode_lengths = []
    
    # For HER: store episode experiences
    episode_experiences = []
    
    # Start training
    for t in range(1, max_timesteps + 1):
        episode_timesteps += 1
        
        # Select action with noise for exploration
        action = policy.select_action(state)
        
        # Perform action
        next_state_dict, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        # Extract observation from dictionary
        next_state = next_state_dict['observation']
        
        # Store data in replay buffer
        replay_buffer.add(state, action, next_state, reward, done)
        
        # For HER: Store episode experiences
        if use_her:
            episode_experiences.append({
                "state": state,
                "action": action,
                "next_state": next_state,
                "reward": reward,
                "done": done,
                "achieved_goal": next_state_dict['achieved_goal'],
                "desired_goal": next_state_dict['desired_goal']
            })
        
        # Update state and episode reward
        state = next_state
        state_dict = next_state_dict
        episode_reward += reward
        
        # Train agent after collecting enough samples
        if t > params['learning_starts']:
            policy.update_parameters(replay_buffer, batch_size=params['batch_size'])
        
        # If end of episode
        if done:
            print(f"Episode {episode_num+1}: Total reward = {episode_reward:.3f}, Length = {episode_timesteps}")
            
            # Store episode data
            training_rewards.append(episode_reward)
            episode_lengths.append(episode_timesteps)
            
            # For HER: Add HER transitions to the replay buffer
            if use_her and len(episode_experiences) > 0:
                replay_buffer.add_her_samples(episode_experiences, 
                                             goal_selection_strategy=her_strategy, 
                                             k=her_k)
                episode_experiences = []  # Reset episode experiences
            
            # Reset environment
            state_dict, _ = env.reset()
            state = state_dict['observation']
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1
        
        # Evaluate periodically
        if (t % eval_freq == 0) or (t == max_timesteps):
            print(f"\nTimestep {t}/{max_timesteps}")
            avg_reward, success_rate = evaluate_policy(policy, eval_env, eval_episodes=10)
            evaluations.append(avg_reward)
            success_rates.append(success_rate)
            
            # Early stopping if good performance
            if success_rate >= 0.9:
                print(f"Reached success rate {success_rate:.2f} >= 0.9. Early stopping.")
                break
            
    # Calculate final performance metrics
    final_success_rate = success_rates[-1] if success_rates else 0
    avg_reward_last_10_episodes = np.mean(training_rewards[-10:]) if training_rewards else 0
    
    # Store training data and parameters
    result = {
        'algorithm': f'SAC_HER_{her_strategy}' if use_her else 'SAC',
        'params': params,
        'rewards': np.array(training_rewards),
        'success_rates': np.array(success_rates),
        'evaluations': np.array(evaluations),
        'episode_lengths': np.array(episode_lengths),
        'final_success_rate': final_success_rate,
        'avg_reward_last_10_episodes': avg_reward_last_10_episodes,
        'env_name': env_name
    }
    
    # Save the best model if success rate is high
    if save_models and final_success_rate > 0.5:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_dir = f"{save_dir}/sac{'_her_' + her_strategy if use_her else ''}_{timestamp}_{final_success_rate:.2f}"
        os.makedirs(model_dir, exist_ok=True)
        policy.save(f"{model_dir}/model")
        
        # Save parameters to a readable format
        with open(f"{model_dir}/params.json", 'w') as f:
            json.dump({k: str(v) if isinstance(v, np.ndarray) else v for k, v in params.items()}, f, indent=4)
    
    env.close()
    eval_env.close()
    
    return result

def train_td3_with_params(params, env_name=ENV_NAME, max_timesteps=30000, eval_freq=5000, 
                           save_models=True, save_dir="./results/td3_random_search", device='cpu',
                           use_her=False, her_strategy="final", her_k=4):
    """
    Train a TD3 agent with the given hyperparameters
    
    Args:
        params: Dictionary of hyperparameters
        env_name: Name of the environment
        max_timesteps: Maximum number of timesteps to train for
        eval_freq: Frequency of evaluation
        save_models: Whether to save models
        save_dir: Directory to save models
        device: Device to use for training
        use_her: Whether to use Hindsight Experience Replay
        her_strategy: HER goal selection strategy ('final', 'future', 'random')
        her_k: Number of goals to sample per transition for HER
        
    Returns:
        result: Dictionary containing training data and performance metrics
    """
    # Create save directory
    if save_models and not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Create environment
    env = gym.make(env_name)
    eval_env = gym.make(env_name)
    
    # Set seeds for reproducibility
    seed = 0
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Get state and action dimensions from environment
    state_dim = env.observation_space['observation'].shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    
    # Initialize policy with hyperparameters
    policy = TD3(
        state_dim=state_dim,
        action_dim=action_dim,
        max_action=max_action,
        device=device,
        hidden_dim=params['hidden_dim'],
        discount=params['gamma'],
        tau=params['tau'],
        policy_noise=params['policy_noise'] * max_action,
        noise_clip=params['noise_clip'] * max_action,
        policy_freq=params['policy_freq']
    )
    
    # Initialize replay buffer - standard or HER
    if use_her:
        replay_buffer = HERReplayBuffer(
            state_dim=state_dim,
            action_dim=action_dim,
            max_size=params['buffer_size'],
            device=device
        )
    else:
        from replay_buffer import ReplayBuffer
        replay_buffer = ReplayBuffer(
            state_dim=state_dim,
            action_dim=action_dim,
            max_size=params['buffer_size'],
            device=device
        )
    
    # Initialize variables
    episode_reward = 0
    episode_timesteps = 0
    episode_num = 0
    state_dict, _ = env.reset(seed=seed)
    state = state_dict['observation']
    
    # Initialize lists for storing data
    evaluations = []
    success_rates = []
    training_rewards = []
    episode_lengths = []
    
    # For HER: store episode experiences
    episode_experiences = []
    
    # Start training
    for t in range(1, max_timesteps + 1):
        episode_timesteps += 1
        
        # Select action with noise for exploration
        action = policy.select_action(state, noise=params['exploration_noise'])
        
        # Perform action
        next_state_dict, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        # Extract observation from dictionary
        next_state = next_state_dict['observation']
        
        # Store data in replay buffer
        replay_buffer.add(state, action, next_state, reward, done)
        
        # For HER: Store episode experiences
        if use_her:
            episode_experiences.append({
                "state": state,
                "action": action,
                "next_state": next_state,
                "reward": reward,
                "done": done,
                "achieved_goal": next_state_dict['achieved_goal'],
                "desired_goal": next_state_dict['desired_goal']
            })
        
        # Update state and episode reward
        state = next_state
        state_dict = next_state_dict
        episode_reward += reward
        
        # Train agent after collecting enough samples
        if t > params['learning_starts']:
            policy.update(replay_buffer, batch_size=params['batch_size'])
        
        # If end of episode
        if done:
            print(f"Episode {episode_num+1}: Total reward = {episode_reward:.3f}, Length = {episode_timesteps}")
            
            # Store episode data
            training_rewards.append(episode_reward)
            episode_lengths.append(episode_timesteps)
            
            # For HER: Add HER transitions to the replay buffer
            if use_her and len(episode_experiences) > 0:
                replay_buffer.add_her_samples(episode_experiences, 
                                             goal_selection_strategy=her_strategy, 
                                             k=her_k)
                episode_experiences = []  # Reset episode experiences
            
            # Reset environment
            state_dict, _ = env.reset()
            state = state_dict['observation']
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1
        
        # Evaluate periodically
        if (t % eval_freq == 0) or (t == max_timesteps):
            print(f"\nTimestep {t}/{max_timesteps}")
            avg_reward, success_rate = evaluate_policy(policy, eval_env, eval_episodes=10)
            evaluations.append(avg_reward)
            success_rates.append(success_rate)
            
            # Early stopping if good performance
            if success_rate >= 0.9:
                print(f"Reached success rate {success_rate:.2f} >= 0.9. Early stopping.")
                break
    
    # Calculate final performance metrics
    final_success_rate = success_rates[-1] if success_rates else 0
    avg_reward_last_10_episodes = np.mean(training_rewards[-10:]) if training_rewards else 0
    
    # Store training data and parameters
    result = {
        'algorithm': f'TD3_HER_{her_strategy}' if use_her else 'TD3',
        'params': params,
        'rewards': np.array(training_rewards),
        'success_rates': np.array(success_rates),
        'evaluations': np.array(evaluations),
        'episode_lengths': np.array(episode_lengths),
        'final_success_rate': final_success_rate,
        'avg_reward_last_10_episodes': avg_reward_last_10_episodes,
        'env_name': env_name
    }
    
    # Save the best model if success rate is high
    if save_models and final_success_rate > 0.5:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_dir = f"{save_dir}/td3{'_her_' + her_strategy if use_her else ''}_{timestamp}_{final_success_rate:.2f}"
        os.makedirs(model_dir, exist_ok=True)
        policy.save(f"{model_dir}/model")
        
        # Save parameters to a readable format
        with open(f"{model_dir}/params.json", 'w') as f:
            json.dump({k: str(v) if isinstance(v, np.ndarray) else v for k, v in params.items()}, f, indent=4)
    
    env.close()
    eval_env.close()
    
    return result

def generate_sac_params():
    """
    Generate random hyperparameters for SAC, optimized for PandaReach environment
    
    Returns:
        params: Dictionary of hyperparameters
    """
    params = {
        'hidden_dim': np.random.choice([64, 128, 256]),
        'lr': np.random.choice([3e-4, 5e-4, 7e-4, 1e-3, 2e-3]),
        'gamma': np.random.uniform(0.9, 0.98),
        'tau': np.random.choice([0.005, 0.01, 0.02, 0.05]),
        'alpha': np.random.uniform(0.01, 0.2),
        'automatic_entropy_tuning': np.random.choice([True, True, False]),
        'batch_size': np.random.choice([32, 64, 128]),
        'buffer_size': int(np.random.choice([5e4, 1e5, 2e5])),
        'learning_starts': np.random.choice([0, 500, 1000])
    }
    return params

def generate_td3_params():
    """
    Generate random hyperparameters for TD3, optimized for PandaReach environment
    
    Returns:
        params: Dictionary of hyperparameters
    """
    params = {
        'hidden_dim': np.random.choice([64, 128, 256]),
        'gamma': np.random.uniform(0.9, 0.98),
        'tau': np.random.choice([0.005, 0.01, 0.02, 0.05]),
        'policy_noise': np.random.uniform(0.1, 0.2),
        'noise_clip': np.random.uniform(0.2, 0.5),
        'policy_freq': np.random.choice([1, 2, 3]),
        'exploration_noise': np.random.uniform(0.05, 0.15),
        'batch_size': np.random.choice([32, 64, 128]),
        'buffer_size': int(np.random.choice([5e4, 1e5, 2e5])),
        'learning_starts': np.random.choice([0, 500, 1000])
    }
    return params

def run_random_search(n_trials=5, max_timesteps=30000, save_dir="./results/random_search", 
                     use_her=False, her_strategy="final", her_k=4, algorithms=None, device='cpu'):
    """
    Run random search for hyperparameters
    
    Args:
        n_trials: Number of trials for each algorithm
        max_timesteps: Maximum number of timesteps to train for each trial
        save_dir: Directory to save results
        use_her: Whether to use Hindsight Experience Replay
        her_strategy: HER goal selection strategy ('final', 'future', 'random')
        her_k: Number of goals to sample per transition for HER
        algorithms: List of algorithms to run (options: 'sac', 'td3', default: ['sac', 'td3'])
        device: Device to use for training
    """
    # Create save directory
    create_directory(save_dir)
    create_directory(f"{save_dir}/sac")
    create_directory(f"{save_dir}/td3")
    
    # Create environment for printing info
    env = gym.make(ENV_NAME)
    print_env_info(env)
    env.close()
    
    # Default algorithms if not specified
    if algorithms is None:
        algorithms = ['sac', 'td3']
    
    # Store results
    sac_results = []
    td3_results = []
    
    # Run random search for SAC
    if 'sac' in algorithms:
        print(f"\n==== Running Random Search for SAC" + (f" with HER ({her_strategy})" if use_her else "") + " ====")
        for i in range(n_trials):
            print(f"\nSAC Trial {i+1}/{n_trials}")
            params = generate_sac_params()
            print(f"Parameters: {params}")
            
            # Train with these parameters
            result = train_sac_with_params(
                params=params,
                max_timesteps=max_timesteps,
                save_dir=f"{save_dir}/sac",
                device=device,
                use_her=use_her,
                her_strategy=her_strategy,
                her_k=her_k
            )
            
            # Store result
            sac_results.append(result)
            
            # Save partial results after each trial
            suffix = f"_her_{her_strategy}" if use_her else ""
            np.save(f"{save_dir}/sac{suffix}_results.npy", sac_results)
            
            print(f"Trial completed. Final success rate: {result['final_success_rate']:.3f}")
    
    # Run random search for TD3
    if 'td3' in algorithms:
        print(f"\n==== Running Random Search for TD3" + (f" with HER ({her_strategy})" if use_her else "") + " ====")
        for i in range(n_trials):
            print(f"\nTD3 Trial {i+1}/{n_trials}")
            params = generate_td3_params()
            print(f"Parameters: {params}")
            
            # Train with these parameters
            result = train_td3_with_params(
                params=params,
                max_timesteps=max_timesteps,
                save_dir=f"{save_dir}/td3",
                device=device,
                use_her=use_her,
                her_strategy=her_strategy,
                her_k=her_k
            )
            
            # Store result
            td3_results.append(result)
            
            # Save partial results after each trial
            suffix = f"_her_{her_strategy}" if use_her else ""
            np.save(f"{save_dir}/td3{suffix}_results.npy", td3_results)
            
            print(f"Trial completed. Final success rate: {result['final_success_rate']:.3f}")
    
    # Analyze results
    if sac_results or td3_results:
        analyze_results(sac_results, td3_results, save_dir, use_her, her_strategy)
    
    return {
        'sac_results': sac_results,
        'td3_results': td3_results
    }

def analyze_results(sac_results, td3_results, save_dir, use_her=False, her_strategy="final"):
    """
    Analyze results from random search
    
    Args:
        sac_results: List of results from SAC trials
        td3_results: List of results from TD3 trials
        save_dir: Directory to save analysis
        use_her: Whether HER was used
        her_strategy: HER strategy used
    """
    # Helper function to convert numpy types to Python standard types
    def convert_to_serializable(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        else:
            return obj
    
    suffix = f"_her_{her_strategy}" if use_her else ""
    
    # Find best parameters for each algorithm
    if sac_results:
        best_sac_idx = np.argmax([result['final_success_rate'] for result in sac_results])
        best_sac = sac_results[best_sac_idx]
        
        # Print best parameters
        print(f"\n==== Best SAC{suffix} Parameters ====")
        print(f"Success Rate: {best_sac['final_success_rate']:.3f}")
        print(f"Parameters: {best_sac['params']}")
    
    if td3_results:
        best_td3_idx = np.argmax([result['final_success_rate'] for result in td3_results])
        best_td3 = td3_results[best_td3_idx]
        
        # Print best parameters
        print(f"\n==== Best TD3{suffix} Parameters ====")
        print(f"Success Rate: {best_td3['final_success_rate']:.3f}")
        print(f"Parameters: {best_td3['params']}")
    
    # Save best parameters
    best_params = {}
    if sac_results:
        best_sac_params = convert_to_serializable(best_sac['params'])
        best_params['sac'] = {
            'success_rate': float(best_sac['final_success_rate']),
            'params': best_sac_params
        }
    
    if td3_results:
        best_td3_params = convert_to_serializable(best_td3['params'])
        best_params['td3'] = {
            'success_rate': float(best_td3['final_success_rate']),
            'params': best_td3_params
        }
    
    with open(f"{save_dir}/best_params{suffix}.json", 'w') as f:
        json.dump(best_params, f, indent=4)
    
    # Plot comparison if both algorithms were run
    if sac_results and td3_results:
        plt.figure(figsize=(18, 6))
        
        # Plot success rates
        plt.subplot(1, 3, 1)
        plt.plot(best_sac['success_rates'], label=f'SAC{suffix}')
        plt.plot(best_td3['success_rates'], label=f'TD3{suffix}')
        plt.xlabel('Evaluation')
        plt.ylabel('Success Rate')
        plt.title('Success Rate Comparison')
        plt.legend()
        plt.grid(True)
        
        # Plot learning curves (episode rewards)
        plt.subplot(1, 3, 2)
        window_size = 10
        weights = np.ones(window_size) / window_size
        
        if len(best_sac['rewards']) >= window_size:
            sac_smooth = np.convolve(best_sac['rewards'], weights, mode='valid')
            plt.plot(sac_smooth, label=f'SAC{suffix}')
        else:
            plt.plot(best_sac['rewards'], label=f'SAC{suffix}')
            
        if len(best_td3['rewards']) >= window_size:
            td3_smooth = np.convolve(best_td3['rewards'], weights, mode='valid')
            plt.plot(td3_smooth, label=f'TD3{suffix}')
        else:
            plt.plot(best_td3['rewards'], label=f'TD3{suffix}')
        
        plt.xlabel('Episode')
        plt.ylabel('Episode Reward (Smoothed)')
        plt.title('Learning Curve Comparison')
        plt.legend()
        plt.grid(True)
        
        # Plot episode lengths
        plt.subplot(1, 3, 3)
        if len(best_sac['episode_lengths']) >= window_size:
            sac_lengths_smooth = np.convolve(best_sac['episode_lengths'], weights, mode='valid')
            plt.plot(sac_lengths_smooth, label=f'SAC{suffix}')
        else:
            plt.plot(best_sac['episode_lengths'], label=f'SAC{suffix}')
            
        if len(best_td3['episode_lengths']) >= window_size:
            td3_lengths_smooth = np.convolve(best_td3['episode_lengths'], weights, mode='valid')
            plt.plot(td3_lengths_smooth, label=f'TD3{suffix}')
        else:
            plt.plot(best_td3['episode_lengths'], label=f'TD3{suffix}')
        
        plt.xlabel('Episode')
        plt.ylabel('Episode Length (Smoothed)')
        plt.title('Episode Length Comparison')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(f"{save_dir}/comparison{suffix}.png")
        plt.show()
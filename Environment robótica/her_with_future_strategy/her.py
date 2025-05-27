import numpy as np
import torch
from replay_buffer import ReplayBuffer

class HERReplayBuffer(ReplayBuffer):
    """
    A simple extension of ReplayBuffer with Hindsight Experience Replay (HER) capabilities.
    """
    def __init__(self, state_dim, action_dim, max_size=1000000, device="cpu"):
        super().__init__(state_dim, action_dim, max_size, device)
        
    def add_her_samples(self, episode_buffer, goal_selection_strategy="future", k=4):
        """
        Add HER samples to the replay buffer.
        
        Args:
            episode_buffer: List of dictionaries containing episode transitions
            goal_selection_strategy: Strategy for selecting goals ('final', 'future', or 'random')
            k: Number of goals to sample for each transition (only used for 'future' strategy)
        """
        # Use the final achieved state as the goal for all transitions in the episode
        if goal_selection_strategy == "final":
            final_achieved_goal = episode_buffer[-1]["achieved_goal"]
            
            for transition in episode_buffer:
                state = transition["state"]
                action = transition["action"]
                next_state = transition["next_state"]
                achieved_goal = transition["achieved_goal"]
                
                # Compute reward for the new goal
                # Distance-based reward: 0 if close enough, -1 otherwise
                her_reward = 0.0 if np.linalg.norm(achieved_goal - final_achieved_goal) < 0.05 else -1.0
                her_done = her_reward == 0.0
                
                # Add the modified transition to the replay buffer
                self.add(state, action, next_state, her_reward, her_done)
        
        # Future strategy: for each timestep, use k random future states as goals
        elif goal_selection_strategy == "future":
            episode_length = len(episode_buffer)
            
            for i, transition in enumerate(episode_buffer):
                state = transition["state"]
                action = transition["action"]
                next_state = transition["next_state"]
                achieved_goal = transition["achieved_goal"]
                
                # Only sample future goals if we have future states to choose from
                if i < episode_length - 1:
                    # Calculate how many future states we can sample from
                    num_available = episode_length - i - 1
                    # Sample min(k, num_available) future timesteps
                    num_samples = min(k, num_available)
                    
                    if num_samples > 0:
                        # Sample k unique future indices if possible
                        if num_samples < num_available:
                            future_indices = np.random.choice(
                                range(i + 1, episode_length), 
                                size=num_samples, 
                                replace=False
                            )
                        else:
                            # If we can't sample k unique indices, just take all future states
                            future_indices = range(i + 1, episode_length)
                        
                        for future_idx in future_indices:
                            future_achieved_goal = episode_buffer[future_idx]["achieved_goal"]
                            
                            # Compute reward for the new goal
                            her_reward = 0.0 if np.linalg.norm(achieved_goal - future_achieved_goal) < 0.05 else -1.0
                            her_done = her_reward == 0.0
                            
                            # Add the modified transition to the replay buffer
                            self.add(state, action, next_state, her_reward, her_done)
        
        # Random strategy: use random states from the episode as goals
        elif goal_selection_strategy == "random":
            episode_length = len(episode_buffer)
            
            for transition in episode_buffer:
                state = transition["state"]
                action = transition["action"]
                next_state = transition["next_state"]
                achieved_goal = transition["achieved_goal"]
                
                # Sample k random states from the episode
                random_indices = np.random.randint(0, episode_length, size=min(k, episode_length))
                
                for random_idx in random_indices:
                    random_achieved_goal = episode_buffer[random_idx]["achieved_goal"]
                    
                    # Compute reward for the new goal
                    her_reward = 0.0 if np.linalg.norm(achieved_goal - random_achieved_goal) < 0.05 else -1.0
                    her_done = her_reward == 0.0
                    
                    # Add the modified transition to the replay buffer
                    self.add(state, action, next_state, her_reward, her_done)

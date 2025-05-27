import numpy as np
import torch
from replay_buffer import ReplayBuffer

class HERReplayBuffer(ReplayBuffer):
    """
    A simple extension of ReplayBuffer with Hindsight Experience Replay (HER) capabilities.
    """
    def __init__(self, state_dim, action_dim, max_size=1000000, device="cpu"):
        super().__init__(state_dim, action_dim, max_size, device)
        
    def add_her_samples(self, episode_buffer, goal_selection_strategy="final"):
        """
        Add HER samples to the replay buffer.
        
        Args:
            episode_buffer: List of dictionaries containing episode transitions
            goal_selection_strategy: Strategy for selecting goals ('final' or 'future')
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
        
        # Future strategy: for each timestep, use a random future state as goal
        elif goal_selection_strategy == "future":
            episode_length = len(episode_buffer)
            
            for i, transition in enumerate(episode_buffer):
                # Select a random future state
                if i < episode_length - 1:  # Only if we have future states
                    future_idx = np.random.randint(i + 1, episode_length)
                    future_achieved_goal = episode_buffer[future_idx]["achieved_goal"]
                    
                    state = transition["state"]
                    action = transition["action"]
                    next_state = transition["next_state"]
                    achieved_goal = transition["achieved_goal"]
                    
                    # Compute reward for the new goal
                    her_reward = 0.0 if np.linalg.norm(achieved_goal - future_achieved_goal) < 0.05 else -1.0
                    her_done = her_reward == 0.0
                    
                    # Add the modified transition to the replay buffer
                    self.add(state, action, next_state, her_reward, her_done)

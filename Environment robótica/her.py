import numpy as np
import torch
from replay_buffer import ReplayBuffer

class HERReplayBuffer(ReplayBuffer):
    """A simple extension of ReplayBuffer with Hindsight Experience Replay (HER) capabilities."""
    
    def __init__(self, state_dim, action_dim, max_size=1000000, device="cpu"):
        super().__init__(state_dim, action_dim, max_size, device)
        
    def add_her_samples(self, episode_buffer, goal_selection_strategy="final", k=4):
        """
        Add HER samples to the replay buffer.
        
        Args:
            episode_buffer: List of dictionaries containing episode transitions
            goal_selection_strategy: Strategy for selecting goals ('final', 'future', or 'random')
            k: Number of goals to sample for each transition (only used for 'future' and 'random')
        """
        if goal_selection_strategy == "final":
            final_achieved_goal = episode_buffer[-1]["achieved_goal"]
            
            for transition in episode_buffer:
                state = transition["state"]
                action = transition["action"]
                next_state = transition["next_state"]
                achieved_goal = transition["achieved_goal"]
                
                # Distance-based reward: 0 if close enough, -1 otherwise
                her_reward = 0.0 if np.linalg.norm(achieved_goal - final_achieved_goal) < 0.05 else -1.0
                her_done = her_reward == 0.0
                
                self.add(state, action, next_state, her_reward, her_done)
        
        elif goal_selection_strategy == "future":
            episode_length = len(episode_buffer)
            
            for i, transition in enumerate(episode_buffer):
                state = transition["state"]
                action = transition["action"]
                next_state = transition["next_state"]
                achieved_goal = transition["achieved_goal"]
                
                if i < episode_length - 1:
                    num_available = episode_length - i - 1
                    num_samples = min(k, num_available)
                    
                    if num_samples > 0:
                        if num_samples < num_available:
                            future_indices = np.random.choice(
                                range(i + 1, episode_length), 
                                size=num_samples, 
                                replace=False
                            )
                        else:
                            future_indices = range(i + 1, episode_length)
                        
                        for future_idx in future_indices:
                            future_achieved_goal = episode_buffer[future_idx]["achieved_goal"]
                            
                            her_reward = 0.0 if np.linalg.norm(achieved_goal - future_achieved_goal) < 0.05 else -1.0
                            her_done = her_reward == 0.0
                            
                            self.add(state, action, next_state, her_reward, her_done)
        
        elif goal_selection_strategy == "random":
            episode_length = len(episode_buffer)
            
            for transition in episode_buffer:
                state = transition["state"]
                action = transition["action"]
                next_state = transition["next_state"]
                achieved_goal = transition["achieved_goal"]
                
                random_indices = np.random.randint(0, episode_length, size=min(k, episode_length))
                
                for random_idx in random_indices:
                    random_achieved_goal = episode_buffer[random_idx]["achieved_goal"]
                    
                    her_reward = 0.0 if np.linalg.norm(achieved_goal - random_achieved_goal) < 0.05 else -1.0
                    her_done = her_reward == 0.0
                    
                    self.add(state, action, next_state, her_reward, her_done)
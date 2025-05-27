import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

# Helper function to initialize network weights
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)

class Actor(nn.Module):
    """
    Actor Network for TD3 algorithm
    """
    def __init__(self, state_dim, action_dim, max_action, hidden_dim=256):
        super(Actor, self).__init__()
        
        self.linear1 = nn.Linear(state_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, action_dim)
        
        self.max_action = max_action
        
        self.apply(weights_init_)
        
    def forward(self, state):
        """
        Forward pass through the network
        """
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        return self.max_action * torch.tanh(self.linear3(x))

class Critic(nn.Module):
    """
    Critic Network for TD3 algorithm (Twin Q-networks)
    """
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Critic, self).__init__()
        
        # Q1 architecture
        self.linear1_q1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.linear2_q1 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3_q1 = nn.Linear(hidden_dim, 1)
        
        # Q2 architecture
        self.linear1_q2 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.linear2_q2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3_q2 = nn.Linear(hidden_dim, 1)
        
        self.apply(weights_init_)
        
    def forward(self, state, action):
        """
        Returns Q-values from both Q-networks
        """
        sa = torch.cat([state, action], 1)
        
        # Q1 forward pass
        q1 = F.relu(self.linear1_q1(sa))
        q1 = F.relu(self.linear2_q1(q1))
        q1 = self.linear3_q1(q1)
        
        # Q2 forward pass
        q2 = F.relu(self.linear1_q2(sa))
        q2 = F.relu(self.linear2_q2(q2))
        q2 = self.linear3_q2(q2)
        
        return q1, q2
    
    def Q1(self, state, action):
        """
        Returns Q-value from first Q-network (for policy update)
        """
        sa = torch.cat([state, action], 1)
        
        q1 = F.relu(self.linear1_q1(sa))
        q1 = F.relu(self.linear2_q1(q1))
        q1 = self.linear3_q1(q1)
        
        return q1

class TD3:
    """
    Twin Delayed DDPG Implementation
    """
    def __init__(
        self, 
        state_dim,
        action_dim,
        max_action,
        device,
        hidden_dim=256,
        discount=0.99,
        tau=0.005,
        policy_noise=0.2,
        noise_clip=0.5,
        policy_freq=2
    ):
        self.device = device
        
        # Initialize actor networks
        self.actor = Actor(state_dim, action_dim, max_action, hidden_dim).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)
        
        # Initialize critic networks
        self.critic = Critic(state_dim, action_dim, hidden_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)
        
        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        
        self.total_it = 0
    
    def select_action(self, state, noise=0.1):
        """
        Select an action from the policy with optional noise for exploration
        """
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        action = self.actor(state).cpu().data.numpy().flatten()
        
        if noise != 0:
            action = action + np.random.normal(0, noise * self.max_action, size=action.shape)
            
        return np.clip(action, -self.max_action, self.max_action)
    
    def update(self, replay_buffer, batch_size=256):
        """
        Update the network parameters
        """
        self.total_it += 1
        
        # Sample a batch from replay buffer
        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)
        
        # Add clipped noise to target actions
        with torch.no_grad():
            # Select action according to target policy and add clipped noise
            noise = torch.randn_like(action) * self.policy_noise
            noise = noise.clamp(-self.noise_clip, self.noise_clip)
            
            next_action = (
                self.actor_target(next_state) + noise
            ).clamp(-self.max_action, self.max_action)
            
            # Compute target Q-value
            target_q1, target_q2 = self.critic_target(next_state, next_action)
            target_q = torch.min(target_q1, target_q2)
            target_q = reward + not_done * self.discount * target_q
            
        # Get current Q estimates
        current_q1, current_q2 = self.critic(state, action)
        
        # Compute critic loss
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
        
        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Delayed policy updates
        policy_loss = None
        if self.total_it % self.policy_freq == 0:
            # Compute actor loss
            policy_loss = -self.critic.Q1(state, self.actor(state)).mean()
            
            # Optimize the actor
            self.actor_optimizer.zero_grad()
            policy_loss.backward()
            self.actor_optimizer.step()
            
            # Soft update target networks
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
                
            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
                
        return {
            'critic_loss': critic_loss.item(),
            'policy_loss': policy_loss.item() if policy_loss is not None else None,
        }
        
    def save(self, filename):
        torch.save(self.critic.state_dict(), filename + "_critic")
        torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")
        
        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")
        
    def load(self, filename):
        self.critic.load_state_dict(torch.load(filename + "_critic"))
        self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
        self.critic_target = copy.deepcopy(self.critic)
        
        self.actor.load_state_dict(torch.load(filename + "_actor"))
        self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
        self.actor_target = copy.deepcopy(self.actor)

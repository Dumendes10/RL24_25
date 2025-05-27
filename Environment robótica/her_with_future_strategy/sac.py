import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Helper function to initialize network weights
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)

class QNetwork(nn.Module):
    """
    Q-Network for SAC algorithm
    """
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(QNetwork, self).__init__()
        
        # Q1 architecture
        self.linear1_q1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.linear2_q1 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3_q1 = nn.Linear(hidden_dim, 1)
        
        # Q2 architecture (for twin Q learning)
        self.linear1_q2 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.linear2_q2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3_q2 = nn.Linear(hidden_dim, 1)
        
        self.apply(weights_init_)
        
    def forward(self, state, action):
        """
        Returns Q-values from both Q-networks
        """
        xu = torch.cat([state, action], 1)
        
        # Q1 forward pass
        x1 = F.relu(self.linear1_q1(xu))
        x1 = F.relu(self.linear2_q1(x1))
        q1 = self.linear3_q1(x1)
        
        # Q2 forward pass
        x2 = F.relu(self.linear1_q2(xu))
        x2 = F.relu(self.linear2_q2(x2))
        q2 = self.linear3_q2(x2)
        
        return q1, q2

class GaussianPolicy(nn.Module):
    """
    Gaussian Policy for SAC algorithm
    """
    def __init__(self, state_dim, action_dim, hidden_dim=256, action_space=None):
        super(GaussianPolicy, self).__init__()
        
        self.linear1 = nn.Linear(state_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        
        self.mean_linear = nn.Linear(hidden_dim, action_dim)
        self.log_std_linear = nn.Linear(hidden_dim, action_dim)
        
        self.apply(weights_init_)
        
        # Action space bounds for scaling
        if action_space is not None:
            self.action_scale = torch.tensor(
                (action_space.high - action_space.low) / 2.0, 
                dtype=torch.float32
            )
            self.action_bias = torch.tensor(
                (action_space.high + action_space.low) / 2.0, 
                dtype=torch.float32
            )
        else:
            self.action_scale = torch.tensor(1.0)
            self.action_bias = torch.tensor(0.0)
            
    def forward(self, state):
        """
        Forward pass through the network
        """
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        
        # Constrain log_std within [LOG_STD_MIN, LOG_STD_MAX]
        LOG_STD_MIN, LOG_STD_MAX = -20, 2
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        
        return mean, log_std
    
    def sample(self, state):
        """
        Sample an action from the policy
        """
        mean, log_std = self.forward(state)
        std = log_std.exp()
        
        # Sample from normal distribution
        normal = torch.distributions.Normal(mean, std)
        
        # Reparameterization trick
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        
        # Scale action to be in correct range
        action = y_t * self.action_scale + self.action_bias
        
        # Calculate log probability
        log_prob = normal.log_prob(x_t)
        
        # Apply tanh squashing correction
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        
        # Calculate mean action (for evaluation)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        
        return action, log_prob, mean

class SAC:
    """
    Soft Actor-Critic implementation
    """
    def __init__(
        self, 
        state_dim,
        action_dim,
        action_space,
        device,
        hidden_dim=256,
        lr=3e-4,
        gamma=0.99,
        tau=0.005,
        alpha=0.2,
        automatic_entropy_tuning=True
    ):
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.device = device
        self.automatic_entropy_tuning = automatic_entropy_tuning
        
        # Initialize critic networks
        self.critic = QNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.critic_target = QNetwork(state_dim, action_dim, hidden_dim).to(device)
        
        # Initialize critic optimizer
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)
        
        # Initialize actor network
        self.policy = GaussianPolicy(state_dim, action_dim, hidden_dim, action_space).to(device)
        
        # Initialize actor optimizer
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        
        # Copy parameters from critic to critic target
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)
            
        # Automatic entropy tuning
        if self.automatic_entropy_tuning:
            self.target_entropy = -torch.prod(torch.Tensor([action_dim]).to(device)).item()
            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=lr)
    
    def select_action(self, state, evaluate=False):
        """
        Select an action from the policy
        """
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        
        if evaluate:
            _, _, action = self.policy.sample(state)
        else:
            action, _, _ = self.policy.sample(state)
            
        return action.detach().cpu().numpy()[0]
    
    def update_parameters(self, replay_buffer, batch_size=256):
        """
        Update the network parameters
        """
        # Sample a batch from replay buffer
        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)
        
        # Update critic
        with torch.no_grad():
            # Sample action from policy
            next_action, next_log_pi, _ = self.policy.sample(next_state)
            
            # Compute target Q-value
            target_q1, target_q2 = self.critic_target(next_state, next_action)
            target_q = torch.min(target_q1, target_q2)
            target_q = reward + not_done * self.gamma * (target_q - self.alpha * next_log_pi)
            
        # Get current Q estimates
        current_q1, current_q2 = self.critic(state, action)
        
        # Compute critic loss
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
        
        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Update actor
        pi, log_pi, _ = self.policy.sample(state)
        
        q1_pi, q2_pi = self.critic(state, pi)
        q_pi = torch.min(q1_pi, q2_pi)
        
        # Compute actor loss
        policy_loss = ((self.alpha * log_pi) - q_pi).mean()
        
        # Optimize the actor
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()
        
        # Update automatic entropy tuning
        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
            
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            
            self.alpha = self.log_alpha.exp()
            
        # Soft update of target networks
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.tau) + param.data * self.tau
            )
            
        return {
            'critic_loss': critic_loss.item(),
            'policy_loss': policy_loss.item(),
            'alpha': self.alpha.item() if self.automatic_entropy_tuning else self.alpha,
        }
        
    def save(self, filename):
        torch.save(self.critic.state_dict(), filename + "_critic")
        torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")
        
        torch.save(self.policy.state_dict(), filename + "_actor")
        torch.save(self.policy_optimizer.state_dict(), filename + "_actor_optimizer")
        
    def load(self, filename):
        self.critic.load_state_dict(torch.load(filename + "_critic"))
        self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
        self.critic_target = copy.deepcopy(self.critic)
        
        self.policy.load_state_dict(torch.load(filename + "_actor"))
        self.policy_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))

import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------ MODEL CONFIGURATION ------------------
NUM_ATOMS = 51       # Options: 21, 31, 51 — more atoms = better distributional precision!!!!!!!!!!!!!!!!! 31
VMIN = -10           # Minimum return
VMAX = 10            # Maximum return
HIDDEN_SIZE = 256    # Hidden layer size in the value & advantage streams
USE_NOISY = True     # Use NoisyLinear instead of Linear for exploration
INPUT_HEIGHT = 84    # Input image height
INPUT_WIDTH = 84     # Input image width
# ---------------------------------------------------------

class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, std_init=0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init

        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))

        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer('bias_epsilon', torch.empty(out_features))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1 / (self.in_features ** 0.5)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / (self.in_features ** 0.5))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / (self.out_features ** 0.5))

    def reset_noise(self):
        device = self.weight_mu.device
        epsilon_in = self._scale_noise(self.in_features, device)
        epsilon_out = self._scale_noise(self.out_features, device)
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def forward(self, x):
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return F.linear(x, weight, bias)

    def _scale_noise(self, size, device):
        x = torch.randn(size, device=device)
        return x.sign() * x.abs().sqrt()


class RainbowDQN(nn.Module):
    def __init__(self, input_shape, num_actions):
        super().__init__()
        self.num_actions = num_actions
        self.num_atoms = NUM_ATOMS
        self.Vmin = VMIN
        self.Vmax = VMAX
        self.use_noisy = USE_NOISY

        c, h, w = input_shape
        assert h == INPUT_HEIGHT and w == INPUT_WIDTH, f"Expected input of shape ({INPUT_HEIGHT}, {INPUT_WIDTH})"

        self.features = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        self.fc_input_dim = self.feature_size(input_shape)
        LinearLayer = NoisyLinear if self.use_noisy else nn.Linear

        self.value_stream = nn.Sequential(
            LinearLayer(self.fc_input_dim, HIDDEN_SIZE),
            nn.ReLU(),
            LinearLayer(HIDDEN_SIZE, self.num_atoms)
        )

        self.advantage_stream = nn.Sequential(
            LinearLayer(self.fc_input_dim, HIDDEN_SIZE),
            nn.ReLU(),
            LinearLayer(HIDDEN_SIZE, self.num_actions * self.num_atoms)
        )

        self.register_buffer("supports", torch.linspace(self.Vmin, self.Vmax, self.num_atoms))
        self.softmax = nn.Softmax(dim=2)

    def feature_size(self, input_shape):
        return self.features(torch.zeros(1, *input_shape)).view(1, -1).size(1)

    def forward(self, x):
        x = x / 255.0
        batch_size = x.size(0)
        x = self.features(x)
        x = x.view(batch_size, -1)

        value = self.value_stream(x).view(batch_size, 1, self.num_atoms)
        advantage = self.advantage_stream(x).view(batch_size, self.num_actions, self.num_atoms)

        q_atoms = value + advantage - advantage.mean(dim=1, keepdim=True)
        probabilities = self.softmax(q_atoms)
        return probabilities

    def reset_noise(self):
        if self.use_noisy:
            for m in self.modules():
                if isinstance(m, NoisyLinear):
                    m.reset_noise()

    def act(self, state):
        with torch.no_grad():
            probabilities = self.forward(state)
            q_values = torch.sum(probabilities * self.supports, dim=2)
            return q_values.argmax(1).item()

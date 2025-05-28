
import torch
import torch.nn as nn
import torch.nn.functional as F

class RainbowDQN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(RainbowDQN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        conv_out_size = self._get_conv_out(input_shape)

        # Dueling DQN heads
        self.fc_value = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

        self.fc_advantage = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(torch.flatten(o, 1).size(1))

    def forward(self, x):
        conv_out = self.conv(x)
        conv_out = torch.flatten(conv_out, 1)
        value = self.fc_value(conv_out)
        advantage = self.fc_advantage(conv_out)
        q_vals = value + advantage - advantage.mean(dim=1, keepdim=True)
        return q_vals

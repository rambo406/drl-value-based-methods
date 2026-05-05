import torch
import torch.nn as nn
import torch.nn.functional as F


class QNetwork(nn.Module):
    """Actor (Policy) Model for Deep Q-Network.

    Maps state vectors to action-value estimates for a Unity ML-Agents
    Banana collector environment (37-dim state, 4 discrete actions).
    """

    def __init__(self, state_size, action_size, seed):
        """Initialize parameters and build model.

        Args:
            state_size (int): Dimension of each state (37).
            action_size (int): Dimension of each action (4).
            seed (int): Random seed for reproducibility.
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, state):
        """Forward pass: map a state to action values.

        Args:
            state (torch.Tensor): Input state tensor of shape (batch, state_size).

        Returns:
            torch.Tensor: Q-values for each action, shape (batch, action_size).
        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

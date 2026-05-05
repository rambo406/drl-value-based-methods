import numpy as np
import random
from collections import namedtuple, deque

import torch
import torch.nn.functional as F
import torch.optim as optim

from model import QNetwork

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64          # minibatch size
GAMMA = 0.99             # discount factor
TAU = 1e-3               # soft update interpolation parameter
LR = 5e-4               # learning rate
UPDATE_EVERY = 4         # how often to update the network

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Agent():
    """DQN Agent that interacts with and learns from a Unity Banana environment.

    Uses experience replay and fixed Q-targets for stable training.
    """

    def __init__(self, state_size, action_size, seed):
        """Initialize an Agent.

        Args:
            state_size (int): Dimension of each state (37).
            action_size (int): Number of discrete actions (4).
            seed (int): Random seed for reproducibility.
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        # Q-Networks (local for training, target for stable TD targets)
        self.qnetwork_local = QNetwork(state_size, action_size, seed).to(device)
        self.qnetwork_target = QNetwork(state_size, action_size, seed).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        # Initialize time step (for triggering learning every UPDATE_EVERY steps)
        self.t_step = 0

    def step(self, state, action, reward, next_state, done):
        """Store experience in replay memory and trigger learning periodically.

        Args:
            state (array_like): Current state.
            action (int): Action taken.
            reward (float): Reward received.
            next_state (array_like): Next state.
            done (bool): Whether the episode ended.
        """
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)

        # Learn every UPDATE_EVERY time steps
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get a random subset and learn
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)

    def act(self, state, eps=0.):
        """Select an action using epsilon-greedy policy.

        Args:
            state (array_like): Current state.
            eps (float): Epsilon for epsilon-greedy action selection.

        Returns:
            int: Chosen action index.
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy()).item()
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma):
        """Update Q-network parameters using a batch of experience tuples.

        Computes TD targets using the target network, calculates MSE loss
        against the local network's predictions, and performs a gradient step.

        Args:
            experiences (tuple): Batch of (states, actions, rewards, next_states, dones).
            gamma (float): Discount factor.
        """
        states, actions, rewards, next_states, dones = experiences

        # Get max predicted Q-values for next states from target model
        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        # Compute TD targets: r + gamma * max_a' Q_target(s', a') * (1 - done)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # Get expected Q-values from local model for the actions taken
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        # Compute MSE loss
        loss = F.mse_loss(Q_expected, Q_targets)

        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Soft update target network: θ_target ← τ*θ_local + (1-τ)*θ_target
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)

    def soft_update(self, local_model, target_model, tau):
        """Soft update target network parameters.

        θ_target = τ*θ_local + (1 - τ)*θ_target

        Args:
            local_model (QNetwork): Source of updated weights.
            target_model (QNetwork): Target network to update.
            tau (float): Interpolation parameter.
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)


class ReplayBuffer():
    """Fixed-size buffer to store experience tuples for replay."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer.

        Args:
            action_size (int): Dimension of each action.
            buffer_size (int): Maximum size of buffer.
            batch_size (int): Size of each training batch.
            seed (int): Random seed.
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience",
                                     field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory.

        Args:
            state (array_like): Current state.
            action (int): Action taken.
            reward (float): Reward received.
            next_state (array_like): Next state.
            done (bool): Whether the episode ended.
        """
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory.

        Returns:
            tuple: (states, actions, rewards, next_states, dones) as torch Tensors.
        """
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)

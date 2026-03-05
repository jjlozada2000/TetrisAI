import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

import random
import numpy as np
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim

from model import TetrisNet
from env import N_ACTIONS


# Hyperparameters

BATCH_SIZE       = 64       # samples per training step
GAMMA            = 0.99     # discount factor — how much future rewards matter
LR               = 1e-3     # learning rate
MEMORY_SIZE      = 50_000   # max experiences stored in replay buffer
EPSILON_START    = 1.0      # start fully random
EPSILON_MIN      = 0.01     # never go below 1% random
EPSILON_DECAY    = 0.9995   # multiply epsilon by this each step
TARGET_UPDATE    = 500      # sync target network every N steps


# Replay Buffer

class ReplayBuffer:
    """
    Stores past experiences as (state, action, reward, next_state, done).
    Training samples random batches from here — this breaks the correlation
    between consecutive steps which would otherwise destabilize training.
    """

    def __init__(self, capacity: int = MEMORY_SIZE):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states,      dtype=np.float32),
            np.array(actions,     dtype=np.int64),
            np.array(rewards,     dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones,       dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


# DQN Agent

class DQNAgent:
    """
    Deep Q-Network agent.

    Parameters
    ----------
    obs_size   : int   — size of the observation vector (45)
    n_actions  : int   — number of possible actions (6)
    device     : str   — 'mps', 'cuda', or 'cpu'
    """

    def __init__(
        self,
        obs_size:  int = 45,
        n_actions: int = N_ACTIONS,
        device:    str = None,
    ):
        # Auto-select best available device (Apple Silicon → MPS)
        if device is None:
            if torch.backends.mps.is_available():
                device = "mps"
            elif torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
        self.device = torch.device(device)
        print(f"Using device: {self.device}")

        self.n_actions = n_actions

        # Two networks
        self.policy_net = TetrisNet(obs_size, n_actions).to(self.device)
        self.target_net = TetrisNet(obs_size, n_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()   # target net is never trained directly

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LR)
        self.loss_fn   = nn.SmoothL1Loss()   # Huber loss — more stable than MSE

        self.memory    = ReplayBuffer(MEMORY_SIZE)

        self.epsilon   = EPSILON_START
        self.steps     = 0
        self.losses    = []

    # Action selection

    def select_action(self, state: np.ndarray) -> int:
        """
        Epsilon-greedy action selection.
        Random action with probability epsilon, best Q-value action otherwise.
        """
        if random.random() < self.epsilon:
            return random.randint(0, self.n_actions - 1)

        with torch.no_grad():
            t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            q_values = self.policy_net(t)
            return int(q_values.argmax().item())

    # Memory

    def remember(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)

    # Training step

    def train_step(self) -> float | None:
        """
        Sample a batch from replay buffer and perform one gradient update.
        Returns loss value, or None if buffer isn't full enough yet.
        """
        if len(self.memory) < BATCH_SIZE:
            return None

        states, actions, rewards, next_states, dones = self.memory.sample(BATCH_SIZE)

        # Move to device
        states      = torch.tensor(states,      device=self.device)
        actions     = torch.tensor(actions,     device=self.device)
        rewards     = torch.tensor(rewards,     device=self.device)
        next_states = torch.tensor(next_states, device=self.device)
        dones       = torch.tensor(dones,       device=self.device)

        # Current Q-values for actions taken
        current_q = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Target Q-values (Bellman equation)
        # Q_target = reward + gamma * max(Q_target_net(next_state)) * (1 - done)
        with torch.no_grad():
            next_q    = self.target_net(next_states).max(1).values
            target_q  = rewards + GAMMA * next_q * (1.0 - dones)

        loss = self.loss_fn(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping — prevents exploding gradients
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        loss_val = loss.item()
        self.losses.append(loss_val)
        return loss_val

    # Epsilon decay

    def decay_epsilon(self):
        self.epsilon = max(EPSILON_MIN, self.epsilon * EPSILON_DECAY)
        self.steps  += 1

    # Target network sync

    def sync_target(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    # Save / load

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            "policy_state_dict": self.policy_net.state_dict(),
            "target_state_dict": self.target_net.state_dict(),
            "optimizer_state":   self.optimizer.state_dict(),
            "epsilon":           self.epsilon,
            "steps":             self.steps,
        }, path)
        print(f"Saved checkpoint → {path}")

    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint["policy_state_dict"])
        self.target_net.load_state_dict(checkpoint["target_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        self.epsilon = checkpoint["epsilon"]
        self.steps   = checkpoint["steps"]
        print(f"Loaded checkpoint ← {path}")


# Sanity check

if __name__ == "__main__":
    print("Running agent sanity check...")
    agent = DQNAgent()

    # Fake some experiences
    for _ in range(100):
        state      = np.random.rand(45).astype(np.float32)
        action     = random.randint(0, N_ACTIONS - 1)
        reward     = random.uniform(-1, 1)
        next_state = np.random.rand(45).astype(np.float32)
        done       = random.random() < 0.1
        agent.remember(state, action, reward, next_state, done)

    loss = agent.train_step()
    print(f"Loss after first batch : {loss}")

    agent.decay_epsilon()
    print(f"Epsilon after decay    : {agent.epsilon:.4f}")

    agent.save("../models/test_checkpoint.pt")
    agent.load("../models/test_checkpoint.pt")

    print("agent.py sanity check passed.")
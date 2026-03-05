"""
agent.py — DQN Agent for Tetris (matching nuno-faria exactly)

Replay memory stores: (current_state, next_state, reward, done)
  - current_state: board features BEFORE the move
  - next_state: board features AFTER the move (the state the agent chose)
  - reward: score gained from this move
  - done: whether the game ended

Training target (computed from replay buffer at train time):
  if done:     target = reward
  if not done: target = reward + discount * model(next_state)

This is simpler than our previous approach — no pre-computing
"best future state", just the state that actually resulted.

Hyperparameters matched to nuno-faria / ChesterHuynh:
  - mem_size: 20000
  - batch_size: 512
  - discount: 0.95
  - lr: 1e-3
  - epsilon: linear decay to 0 at 75% of episodes
  - epochs: 1 (train once per episode)
"""

import os, sys, random
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim

from model import TetrisNet


class DQNAgent:
    def __init__(
        self,
        state_size=4,
        discount=0.95,
        replay_size=20000,
        replay_start_size=None,
        epsilon=1.0,
        epsilon_min=0.0,
        epsilon_stop_episode=1500,
        batch_size=512,
        epochs=1,
        device=None,
    ):
        if device is None:
            if torch.backends.mps.is_available():
                device = "mps"
            elif torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
        self.device = torch.device(device)
        print(f"Using device: {self.device}")

        self.state_size = state_size
        self.discount = discount
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_stop_episode = epsilon_stop_episode
        self.epsilon_decay = (epsilon - epsilon_min) / epsilon_stop_episode
        self.batch_size = batch_size
        self.epochs = epochs
        self.episode = 0

        self.replay_start_size = replay_start_size or (batch_size * 2)
        self.memory = deque(maxlen=replay_size)

        self.model = TetrisNet(state_size).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        self.loss_fn = nn.MSELoss()

    def best_state(self, states):
        """
        Given list of (state_array, placement_info), pick the best one.
        Epsilon-greedy: random with probability epsilon, else network's best.
        Returns (index, state_array).
        """
        if not states:
            return 0, np.zeros(self.state_size, dtype=np.float32)

        if random.random() < self.epsilon:
            idx = random.randint(0, len(states) - 1)
            return idx, states[idx][0]

        batch = np.array([s[0] for s in states], dtype=np.float32)
        with torch.no_grad():
            self.model.eval()
            scores = self.model(torch.tensor(batch, device=self.device))
            scores = scores.squeeze(1).cpu().numpy()
            self.model.train()

        idx = int(np.argmax(scores))
        return idx, states[idx][0]

    def add_to_memory(self, current_state, next_state, reward, done):
        """Store transition. current_state and next_state are numpy arrays."""
        self.memory.append((current_state, next_state, reward, done))

    def train(self):
        """
        Train once on a batch from replay memory.
        Called ONCE PER EPISODE.
        """
        if len(self.memory) < self.replay_start_size:
            return None

        total_loss = 0.0
        for _ in range(self.epochs):
            batch = random.sample(self.memory, min(self.batch_size, len(self.memory)))
            current_states, next_states, rewards, dones = zip(*batch)

            current_states = torch.tensor(np.array(current_states, dtype=np.float32), device=self.device)
            next_states    = torch.tensor(np.array(next_states, dtype=np.float32), device=self.device)
            rewards        = torch.tensor(np.array(rewards, dtype=np.float32), device=self.device)
            dones          = torch.tensor(np.array(dones, dtype=np.float32), device=self.device)

            # Target: reward + discount * V(next_state) if not done, else just reward
            self.model.eval()
            with torch.no_grad():
                next_v = self.model(next_states).squeeze(1)
            self.model.train()

            targets = rewards + self.discount * next_v * (1.0 - dones)

            # Prediction: V(current_state)
            predictions = self.model(current_states).squeeze(1)

            loss = self.loss_fn(predictions, targets)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / self.epochs

    def decay_epsilon(self):
        self.episode += 1
        self.epsilon = max(self.epsilon_min, self.epsilon - self.epsilon_decay)

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
            "episode": self.episode,
        }, path)

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.epsilon = ckpt["epsilon"]
        self.episode = ckpt.get("episode", 0)
        print(f"Loaded ← {path} (ep={self.episode}, ε={self.epsilon:.3f})")


if __name__ == "__main__":
    print("Agent sanity check...")
    agent = DQNAgent()
    for _ in range(1100):
        s = np.random.rand(4).astype(np.float32)
        agent.add_to_memory(s, np.random.rand(4).astype(np.float32), 1.0, False)
    loss = agent.train()
    print(f"Loss: {loss:.4f}")
    print("OK")
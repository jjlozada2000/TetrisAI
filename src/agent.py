"""
agent.py — DQN Agent for Tetris (Board-Evaluation Approach)

Instead of Q(state) → 40 action values, this agent:
  1. Gets all valid placements from the environment
  2. For each placement, gets the resulting board features
  3. Feeds each feature vector through the network → V(next_state)
  4. Picks the placement with the highest V(next_state)

Training uses the standard DQN update:
  V(state) = reward + gamma * V(best_next_state)

This is much easier to learn because the network only outputs 1 value
and evaluates concrete board states rather than abstract action indices.
"""

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
from env import N_FEATURES


# ── Hyperparameters ───────────────────────────────────────────────────────────

BATCH_SIZE       = 512      # larger batches — more stable gradients
GAMMA            = 0.99     # higher discount — long-term planning matters in Tetris
LR               = 1e-3     # can be higher since network is smaller (31→64→64→1)
MEMORY_SIZE      = 30_000   # replay buffer size
EPSILON_START    = 1.0      # start fully random
EPSILON_MIN      = 0.001    # very low — the network should handle exploration via scoring
EPSILON_DECAY    = 0.999    # per episode
TARGET_UPDATE    = 500      # sync target network every N episodes
MIN_REPLAY_SIZE  = 2000     # fill buffer before training


# ── Replay Buffer ─────────────────────────────────────────────────────────────

class ReplayBuffer:
    """
    Stores (state_features, reward, next_state_features, done).
    state_features: the features of the board state the agent chose.
    next_state_features: the features of the best next state (after next piece).
    """

    def __init__(self, capacity: int = MEMORY_SIZE):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, reward, next_state, done):
        self.buffer.append((state, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states,      dtype=np.float32),
            np.array(rewards,     dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones,       dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


# ── DQN Agent ─────────────────────────────────────────────────────────────────

class DQNAgent:
    """
    Board-evaluation DQN agent.

    Evaluates each possible placement by scoring the resulting board state.
    Picks the placement with the highest score (with epsilon-greedy exploration).
    """

    def __init__(self, device: str = None):
        if device is None:
            if torch.backends.mps.is_available():
                device = "mps"
            elif torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
        self.device = torch.device(device)
        print(f"Using device: {self.device}")

        # Two networks — policy learns, target provides stable targets
        self.policy_net = TetrisNet(N_FEATURES).to(self.device)
        self.target_net = TetrisNet(N_FEATURES).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LR)
        self.loss_fn   = nn.MSELoss()

        self.memory    = ReplayBuffer(MEMORY_SIZE)

        self.epsilon   = EPSILON_START
        self.episodes  = 0
        self.losses    = []

    # ── Action selection ──────────────────────────────────────────────────────

    def select_best(self, next_states):
        """
        Given a list of (features, placement_info) from env.get_next_states(),
        return the index of the best placement.

        With probability epsilon, picks randomly (exploration).
        Otherwise, scores each state with the policy network and picks the best.
        """
        if not next_states:
            return 0

        if random.random() < self.epsilon:
            return random.randint(0, len(next_states) - 1)

        # Score all candidate states in one batch
        feature_batch = np.array([s[0] for s in next_states], dtype=np.float32)
        with torch.no_grad():
            self.policy_net.eval()
            t = torch.tensor(feature_batch, device=self.device)
            scores = self.policy_net(t).squeeze(1).cpu().numpy()
            self.policy_net.train()

        return int(np.argmax(scores))

    def score_states(self, feature_batch, use_target=False):
        """Score a batch of feature vectors. Returns numpy array of scores."""
        net = self.target_net if use_target else self.policy_net
        with torch.no_grad():
            net.eval()
            t = torch.tensor(feature_batch, device=self.device)
            scores = net(t).squeeze(1).cpu().numpy()
        return scores

    # ── Memory ────────────────────────────────────────────────────────────────

    def remember(self, state_features, reward, next_state_features, done):
        self.memory.push(state_features, reward, next_state_features, done)

    # ── Training step ─────────────────────────────────────────────────────────

    def train_step(self) -> float | None:
        """
        Sample a batch and train the value network.
        Target: V(s) = reward + gamma * V_target(best_next_state)
        """
        if len(self.memory) < MIN_REPLAY_SIZE:
            return None

        states, rewards, next_states, dones = self.memory.sample(BATCH_SIZE)

        states      = torch.tensor(states,      device=self.device)
        rewards     = torch.tensor(rewards,     device=self.device)
        next_states = torch.tensor(next_states, device=self.device)
        dones       = torch.tensor(dones,       device=self.device)

        # Current V(state)
        self.policy_net.train()
        current_v = self.policy_net(states).squeeze(1)

        # Target V(state) = reward + gamma * V_target(next_state) * (1 - done)
        with torch.no_grad():
            next_v   = self.target_net(next_states).squeeze(1)
            target_v = rewards + GAMMA * next_v * (1.0 - dones)

        loss = self.loss_fn(current_v, target_v)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        loss_val = loss.item()
        self.losses.append(loss_val)
        return loss_val

    # ── Epsilon decay ─────────────────────────────────────────────────────────

    def decay_epsilon(self):
        self.epsilon = max(EPSILON_MIN, self.epsilon * EPSILON_DECAY)
        self.episodes += 1

    # ── Target network sync ───────────────────────────────────────────────────

    def sync_target(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    # ── Save / load ───────────────────────────────────────────────────────────

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            "policy_state_dict": self.policy_net.state_dict(),
            "target_state_dict": self.target_net.state_dict(),
            "optimizer_state":   self.optimizer.state_dict(),
            "epsilon":           self.epsilon,
            "episodes":          self.episodes,
        }, path)
        print(f"Saved checkpoint → {path}")

    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint["policy_state_dict"])
        self.target_net.load_state_dict(checkpoint["target_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        self.epsilon  = checkpoint["epsilon"]
        self.episodes = checkpoint.get("episodes", 0)
        print(f"Loaded checkpoint ← {path}")


# ── Sanity check ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Running agent sanity check...")
    agent = DQNAgent()

    # Fake experiences
    for _ in range(2100):
        state      = np.random.rand(N_FEATURES).astype(np.float32)
        reward     = random.uniform(0, 2)
        next_state = np.random.rand(N_FEATURES).astype(np.float32)
        done       = random.random() < 0.05
        agent.remember(state, reward, next_state, done)

    loss = agent.train_step()
    print(f"Loss after first batch : {loss}")

    agent.decay_epsilon()
    print(f"Epsilon after decay    : {agent.epsilon:.4f}")

    agent.save("../models/test_checkpoint.pt")
    agent.load("../models/test_checkpoint.pt")

    print("agent.py sanity check passed.")
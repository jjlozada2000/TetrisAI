"""
train.py — DQN Training for Tetris

Closely follows vietnh1009/Tetris-deep-Q-learning-pytorch (522 stars).

Key design (proven to achieve infinite play):
  - replay_memory stores (state, reward, next_state, done) tuples
  - Each step: get_next_states(), pick best (ε-greedy), execute, store transition
  - Each episode end: sample batch, compute targets AT TRAIN TIME:
      target = reward                              if done
      target = reward + gamma * model(next_state)  if not done
  - Train model to predict: model(state) ≈ target
  
Hyperparameters (vietnh1009):
  - batch_size: 512
  - lr: 1e-3
  - gamma: 0.99
  - replay_memory_size: 30000
  - initial_epsilon: 1.0
  - final_epsilon: 1e-3
  - num_decay_epochs: 2000
  - num_epochs: 3000
"""

import os, sys, argparse, time, datetime, random
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from collections import deque

import torch
import torch.nn as nn

from env import TetrisEnv
from model import DeepQNetwork

# ── Hyperparameters (matching vietnh1009) ─────────────────────────────────────

BATCH_SIZE         = 512
LR                 = 1e-3
GAMMA              = 0.99
REPLAY_SIZE        = 30000
INITIAL_EPSILON    = 1.0
FINAL_EPSILON      = 1e-3
NUM_DECAY_EPOCHS   = 2000
NUM_EPOCHS         = 3000
SAVE_INTERVAL      = 500
LOG_EVERY          = 10

CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
RUNS_DIR       = os.path.join(os.path.dirname(__file__), "..", "runs")
BEST_PATH      = os.path.join(CHECKPOINT_DIR, "best.pt")
LATEST_PATH    = os.path.join(CHECKPOINT_DIR, "latest.pt")


def train(render=False, resume=False):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(RUNS_DIR, exist_ok=True)

    # Device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # Environment and model
    env = TetrisEnv(render=False)
    model = DeepQNetwork().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()

    # Replay memory
    replay_memory = deque(maxlen=REPLAY_SIZE)

    # Epsilon schedule (linear decay)
    epsilon = INITIAL_EPSILON

    # Resume
    start_epoch = 0
    if resume and os.path.exists(LATEST_PATH):
        ckpt = torch.load(LATEST_PATH, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        epsilon = ckpt.get("epsilon", INITIAL_EPSILON)
        start_epoch = ckpt.get("epoch", 0)
        print(f"Resumed from epoch {start_epoch}, ε={epsilon:.4f}")

    # Logging
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(RUNS_DIR, f"train_{ts}.log")
    log_file = open(log_path, "w", buffering=1)
    log_file.write("epoch,score,lines,pieces,reward,epsilon,loss\n")
    print(f"Logging to: {log_path}")
    print(f"Training for {NUM_EPOCHS} epochs, ε decay ends at {NUM_DECAY_EPOCHS}\n")

    best_score = 0
    recent_scores = []

    for epoch in range(start_epoch, NUM_EPOCHS):
        current_state = env.reset()
        done = False
        ep_reward = 0
        steps = 0
        t0 = time.time()

        # ── Play one full game ────────────────────────────────────────────────
        while not done:
            next_states = env.get_next_states()
            if not next_states:
                break

            # Epsilon-greedy action selection
            if random.random() < epsilon:
                action_key = random.choice(list(next_states.keys()))
            else:
                # Score all next states, pick the best
                action_keys = list(next_states.keys())
                state_arrays = np.array([next_states[k][0] for k in action_keys], dtype=np.float32)
                with torch.no_grad():
                    model.eval()
                    predictions = model(torch.tensor(state_arrays, device=device)).squeeze(1)
                    model.train()
                best_idx = torch.argmax(predictions).item()
                action_key = action_keys[best_idx]

            # The state we're about to transition to
            next_state = next_states[action_key][0]

            # Execute
            _, reward, done, info = env.step(action_key, render=(render and epoch % 50 == 0))
            ep_reward += reward
            steps += 1

            # Store transition: (current_state, reward, next_state, done)
            replay_memory.append((current_state, reward, next_state, done))

            # Advance
            current_state = next_state

        # ── Train at end of episode ───────────────────────────────────────────
        loss_val = 0.0
        if len(replay_memory) >= BATCH_SIZE // 2:
            batch = random.sample(replay_memory, min(BATCH_SIZE, len(replay_memory)))
            
            batch_states = torch.tensor(
                np.array([t[0] for t in batch], dtype=np.float32), device=device)
            batch_rewards = torch.tensor(
                np.array([t[1] for t in batch], dtype=np.float32), device=device)
            batch_next_states = torch.tensor(
                np.array([t[2] for t in batch], dtype=np.float32), device=device)
            batch_dones = torch.tensor(
                np.array([t[3] for t in batch], dtype=np.float32), device=device)

            # Compute targets AT TRAIN TIME (not pre-computed)
            model.eval()
            with torch.no_grad():
                next_predictions = model(batch_next_states).squeeze(1)
            model.train()

            # target = reward if done, else reward + gamma * V(next_state)
            targets = batch_rewards + GAMMA * next_predictions * (1 - batch_dones)

            # Predict current state values
            predictions = model(batch_states).squeeze(1)

            loss = loss_fn(predictions, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_val = loss.item()

        # ── Epsilon decay (linear) ────────────────────────────────────────────
        epsilon = max(FINAL_EPSILON,
                      INITIAL_EPSILON - (INITIAL_EPSILON - FINAL_EPSILON) * (epoch + 1) / NUM_DECAY_EPOCHS)

        # ── Logging ───────────────────────────────────────────────────────────
        score = info["score"]
        lines = info["lines"]
        pieces = info.get("pieces", 0)

        recent_scores.append(score)
        if len(recent_scores) > 50:
            recent_scores.pop(0)

        if score > best_score:
            best_score = score
            torch.save(model.state_dict(), BEST_PATH)

        if (epoch + 1) % SAVE_INTERVAL == 0:
            torch.save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epsilon": epsilon,
                "epoch": epoch + 1,
            }, os.path.join(CHECKPOINT_DIR, f"checkpoint_ep{epoch+1}.pt"))

        if (epoch + 1) % 50 == 0:
            torch.save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epsilon": epsilon,
                "epoch": epoch + 1,
            }, LATEST_PATH)

        log_file.write(f"{epoch+1},{score},{lines},{pieces},{ep_reward:.1f},{epsilon:.4f},{loss_val:.5f}\n")

        if (epoch + 1) % LOG_EVERY == 0:
            dt = time.time() - t0
            bar_len = 20
            filled = int(bar_len * (epoch + 1) / NUM_EPOCHS)
            bar = "█" * filled + "░" * (bar_len - filled)
            pct = 100 * (epoch + 1) / NUM_EPOCHS
            avg50 = np.mean(recent_scores) if recent_scores else 0
            print(
                f"[{bar}] {pct:5.1f}%  "
                f"ep={epoch+1:>5}  score={score:>7}  best={best_score:>7}  "
                f"avg50={avg50:>7.0f}  lines={lines:>4}  pieces={pieces:>4}  "
                f"ε={epsilon:.3f}  loss={loss_val:.4f}  t={dt:.1f}s"
            )

    # Save final
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epsilon": epsilon,
        "epoch": NUM_EPOCHS,
    }, LATEST_PATH)
    torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, "final.pt"))

    print(f"\nDone. Best score: {best_score}")
    log_file.close()
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    try:
        train(render=args.render, resume=args.resume)
    except KeyboardInterrupt:
        print("\nInterrupted.")
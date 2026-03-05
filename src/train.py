"""
train.py — DQN Training Loop for Tetris (Board-Evaluation)

The agent evaluates board states, not actions. Each step:
  1. Get all valid placements and their resulting board features
  2. Pick the best one (or random during exploration)
  3. Execute it, get reward
  4. Store (chosen_state_features, reward, best_next_state_features, done)
  5. Train the value network

Usage:
    python3 src/train.py                  # headless, fast
    python3 src/train.py --render         # watch it play in real time
    python3 src/train.py --resume         # continue from last checkpoint
"""

import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

import argparse
import time
import datetime
import numpy as np

from env   import TetrisEnv
from agent import DQNAgent, TARGET_UPDATE

# ── Config ────────────────────────────────────────────────────────────────────

EPISODES         = 3_000     # converges much faster with this architecture
SAVE_EVERY       = 500       # save numbered checkpoint every N episodes
SAVE_LATEST      = 50        # save latest every N episodes
LOG_EVERY        = 10        # log summary every N episodes
RENDER_FPS       = 30
CHECKPOINT_DIR   = os.path.join(os.path.dirname(__file__), "..", "models")
RUNS_DIR         = os.path.join(os.path.dirname(__file__), "..", "runs")
BEST_PATH        = os.path.join(CHECKPOINT_DIR, "best.pt")
LATEST_PATH      = os.path.join(CHECKPOINT_DIR, "latest.pt")


# ── Logger ────────────────────────────────────────────────────────────────────

class Logger:
    def __init__(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.path = path
        self.file = open(path, "w", buffering=1)
        self.file.write("episode,score,lines,pieces,reward_total,epsilon,loss_avg,duration_s\n")
        print(f"Logging to: {path}\n")

    def write(self, ep, score, lines, pieces, reward, epsilon, loss_avg, duration):
        self.file.write(
            f"{ep},{score},{lines},{pieces},{reward:.3f},"
            f"{epsilon:.4f},{loss_avg:.5f},{duration:.1f}\n"
        )

    def close(self):
        self.file.close()


def timestamp():
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


def print_progress(ep, total, score, best, lines, pieces, epsilon, loss_avg, duration):
    bar_len  = 20
    filled   = int(bar_len * ep / total)
    bar      = "█" * filled + "░" * (bar_len - filled)
    pct      = 100 * ep / total
    print(
        f"[{bar}] {pct:5.1f}%  "
        f"ep={ep:>5}  score={score:>6}  best={best:>6}  "
        f"lines={lines:>4}  pieces={pieces:>3}  ε={epsilon:.3f}  "
        f"loss={loss_avg:.4f}  t={duration:.1f}s"
    )


# ── Training loop ─────────────────────────────────────────────────────────────

def train(render: bool = False, resume: bool = False):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(RUNS_DIR, exist_ok=True)

    log_path = os.path.join(RUNS_DIR, f"train_{timestamp()}.log")
    logger   = Logger(log_path)

    env   = TetrisEnv(render=render, render_fps=RENDER_FPS)
    agent = DQNAgent()

    if resume and os.path.exists(LATEST_PATH):
        agent.load(LATEST_PATH)
        print("Resumed from latest checkpoint.")

    best_score = 0

    print(f"Training for {EPISODES} episodes | render={'on' if render else 'off'}\n")

    for ep in range(1, EPISODES + 1):
        env.reset()
        done         = False
        ep_reward    = 0.0
        ep_losses    = []
        ep_start     = time.time()

        while not done:
            # Get all valid placements and their resulting board features
            next_states = env.get_next_states()

            if not next_states:
                break

            # Pick the best placement (or random during exploration)
            chosen_idx = agent.select_best(next_states)
            chosen_features = next_states[chosen_idx][0]

            # Execute the placement
            _, reward, done, info = env.step(chosen_idx)
            ep_reward += reward

            # For the replay buffer, we need the best next state features
            # (i.e., the best state AFTER the next piece is placed)
            if not done:
                future_states = env.get_next_states()
                if future_states:
                    # Use target network to find best next state
                    future_features = np.array([s[0] for s in future_states], dtype=np.float32)
                    future_scores = agent.score_states(future_features, use_target=True)
                    best_next_features = future_features[np.argmax(future_scores)]
                else:
                    best_next_features = chosen_features  # terminal
                    done = True
            else:
                best_next_features = chosen_features  # terminal

            agent.remember(chosen_features, reward, best_next_features, done)

            # Train
            loss = agent.train_step()
            if loss is not None:
                ep_losses.append(loss)

        # ── Episode complete ──────────────────────────────────────────────────
        agent.decay_epsilon()

        # Sync target network periodically
        if ep % TARGET_UPDATE == 0:
            agent.sync_target()

        score    = info["score"]
        lines    = info["lines"]
        pieces   = info.get("pieces", 0)
        duration = time.time() - ep_start
        loss_avg = float(np.mean(ep_losses)) if ep_losses else 0.0

        if score > best_score:
            best_score = score
            agent.save(BEST_PATH)

        if ep % SAVE_EVERY == 0:
            ckpt_path = os.path.join(CHECKPOINT_DIR, f"checkpoint_ep{ep}.pt")
            agent.save(ckpt_path)

        if ep % SAVE_LATEST == 0:
            agent.save(LATEST_PATH)

        logger.write(ep, score, lines, pieces, ep_reward,
                     agent.epsilon, loss_avg, duration)

        if ep % LOG_EVERY == 0:
            print_progress(ep, EPISODES, score, best_score,
                           lines, pieces, agent.epsilon, loss_avg, duration)

    # ── Done ──────────────────────────────────────────────────────────────────
    agent.save(LATEST_PATH)
    print(f"\nTraining complete. Best score: {best_score}")
    print(f"Log saved to: {log_path}")
    logger.close()
    env.close()


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the Tetris DQN agent.")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    try:
        train(render=args.render, resume=args.resume)
    except KeyboardInterrupt:
        print("\nTraining interrupted. Progress saved to models/latest.pt")
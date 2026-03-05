"""
train.py — DQN Training Loop for Tetris

Usage:
    python3 src/train.py                  # headless, fast
    python3 src/train.py --render         # watch it play in real time
    python3 src/train.py --resume         # continue from last checkpoint
    python3 src/train.py --render --resume

Logs are saved to: runs/train_<timestamp>.log
Checkpoints are saved to: models/checkpoint_ep<N>.pt
                           models/best.pt  (best score so far)
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

EPISODES         = 5_000     # total training episodes
SAVE_EVERY       = 250       # save checkpoint every N episodes
LOG_EVERY        = 10        # log summary every N episodes
RENDER_FPS       = 30        # FPS when --render is on
CHECKPOINT_DIR   = os.path.join(os.path.dirname(__file__), "..", "models")
RUNS_DIR         = os.path.join(os.path.dirname(__file__), "..", "runs")
BEST_PATH        = os.path.join(CHECKPOINT_DIR, "best.pt")
LATEST_PATH      = os.path.join(CHECKPOINT_DIR, "latest.pt")


# ── Logger ────────────────────────────────────────────────────────────────────

class Logger:
    def __init__(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.path = path
        self.file = open(path, "w", buffering=1)  # line-buffered
        header = (
            "episode,score,lines,pieces,reward_total,"
            "epsilon,loss_avg,duration_s\n"
        )
        self.file.write(header)
        print(f"Logging to: {path}\n")

    def write(self, ep, score, lines, pieces, reward, epsilon, loss_avg, duration):
        row = (
            f"{ep},{score},{lines},{pieces},{reward:.3f},"
            f"{epsilon:.4f},{loss_avg:.5f},{duration:.1f}\n"
        )
        self.file.write(row)

    def close(self):
        self.file.close()


def timestamp():
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


def print_progress(ep, total, score, best, lines, epsilon, loss_avg, duration):
    bar_len  = 20
    filled   = int(bar_len * ep / total)
    bar      = "█" * filled + "░" * (bar_len - filled)
    pct      = 100 * ep / total
    print(
        f"[{bar}] {pct:5.1f}%  "
        f"ep={ep:>5}  score={score:>6}  best={best:>6}  "
        f"lines={lines:>4}  ε={epsilon:.3f}  "
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

    start_ep = 1
    if resume and os.path.exists(LATEST_PATH):
        agent.load(LATEST_PATH)
        print("Resumed from latest checkpoint.")

    best_score = 0
    step_count = 0   # global step counter across all episodes

    print(f"Training for {EPISODES} episodes | render={'on' if render else 'off'}\n")

    for ep in range(start_ep, EPISODES + 1):
        obs, _       = env.reset()
        done         = False
        ep_reward    = 0.0
        ep_losses    = []
        ep_start     = time.time()

        while not done:
            action = agent.select_action(obs)
            next_obs, reward, done, info = env.step(action)

            agent.remember(obs, action, reward, next_obs, done)

            loss = agent.train_step()
            if loss is not None:
                ep_losses.append(loss)

            agent.decay_epsilon()
            step_count += 1

            # Sync target network
            if step_count % TARGET_UPDATE == 0:
                agent.sync_target()

            obs        = next_obs
            ep_reward += reward

        # ── Episode complete ──────────────────────────────────────────────────
        score    = info["score"]
        lines    = info["lines"]
        pieces   = info.get("pieces", 0)
        duration = time.time() - ep_start
        loss_avg = float(np.mean(ep_losses)) if ep_losses else 0.0

        # Save best
        if score > best_score:
            best_score = score
            agent.save(BEST_PATH)

        # Save checkpoint
        if ep % SAVE_EVERY == 0:
            ckpt_path = os.path.join(CHECKPOINT_DIR, f"checkpoint_ep{ep}.pt")
            agent.save(ckpt_path)

        # Always keep a latest checkpoint for resuming
        agent.save(LATEST_PATH)

        # Log to file
        logger.write(ep, score, lines, pieces, ep_reward,
                     agent.epsilon, loss_avg, duration)

        # Print to terminal every LOG_EVERY episodes
        if ep % LOG_EVERY == 0:
            print_progress(ep, EPISODES, score, best_score,
                           lines, agent.epsilon, loss_avg, duration)

    # ── Done ──────────────────────────────────────────────────────────────────
    print(f"\nTraining complete. Best score: {best_score}")
    print(f"Log saved to: {log_path}")
    logger.close()
    env.close()


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the Tetris DQN agent.")
    parser.add_argument("--render", action="store_true",
                        help="Render the game window during training.")
    parser.add_argument("--resume", action="store_true",
                        help="Resume training from the latest checkpoint.")
    args = parser.parse_args()

    try:
        train(render=args.render, resume=args.resume)
    except KeyboardInterrupt:
        print("\nTraining interrupted. Progress saved to models/latest.pt")
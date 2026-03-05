"""
watch.py — Watch the current best agent play without interrupting training.

Usage:
    python3 src/watch.py               # watch 5 games at normal speed
    python3 src/watch.py --games 10    # watch 10 games
    python3 src/watch.py --fps 10      # slow it down to see decisions clearly
"""

import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

import argparse
import numpy as np

from env   import TetrisEnv
from agent import DQNAgent

CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
BEST_PATH      = os.path.join(CHECKPOINT_DIR, "best.pt")
LATEST_PATH    = os.path.join(CHECKPOINT_DIR, "latest.pt")


def watch(games: int = 5, fps: int = 30):
    if os.path.exists(BEST_PATH):
        ckpt = BEST_PATH
    elif os.path.exists(LATEST_PATH):
        ckpt = LATEST_PATH
    else:
        print("No checkpoint found. Train the agent first with: python3 src/train.py")
        return

    print(f"Loading checkpoint: {ckpt}")
    agent = DQNAgent()
    agent.load(ckpt)
    agent.epsilon = 0.0   # pure exploitation

    env = TetrisEnv(render=True, render_fps=fps)

    for game in range(1, games + 1):
        env.reset()
        done = False
        total_reward = 0.0

        while not done:
            next_states = env.get_next_states()
            if not next_states:
                break
            idx = agent.select_best(next_states)
            _, reward, done, info = env.step(idx)
            total_reward += reward

        print(
            f"Game {game}/{games} — "
            f"score={info['score']}  lines={info['lines']}  "
            f"level={info['level']}  pieces={info['pieces']}  "
            f"reward={total_reward:.1f}"
        )

    env.close()
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Watch the trained Tetris agent play.")
    parser.add_argument("--games", type=int, default=5)
    parser.add_argument("--fps",   type=int, default=30)
    args = parser.parse_args()

    try:
        watch(games=args.games, fps=args.fps)
    except KeyboardInterrupt:
        print("\nStopped.")
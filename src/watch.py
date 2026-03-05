"""
watch.py — Watch the trained agent play.

Usage:
    python3 src/watch.py
    python3 src/watch.py --games 10
    python3 src/watch.py --fps 10
"""

import os, sys, argparse
sys.path.insert(0, os.path.dirname(__file__))

from env import TetrisEnv
from agent import DQNAgent

CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
BEST_PATH      = os.path.join(CHECKPOINT_DIR, "best.pt")
LATEST_PATH    = os.path.join(CHECKPOINT_DIR, "latest.pt")


def watch(games=5, fps=30):
    ckpt = BEST_PATH if os.path.exists(BEST_PATH) else LATEST_PATH
    if not os.path.exists(ckpt):
        print("No checkpoint found. Train first: python3 src/train.py")
        return

    agent = DQNAgent()
    agent.load(ckpt)
    agent.epsilon = 0.0

    env = TetrisEnv(render=True, render_fps=fps)

    for g in range(1, games + 1):
        env.reset()
        done = False
        total_reward = 0

        while not done:
            states = env.get_next_states()
            if not states:
                break
            idx, _ = agent.best_state(states)
            reward, done, info = env.step(idx)
            total_reward += reward

        print(f"Game {g}/{games} — score={info['score']}  lines={info['lines']}  "
              f"pieces={info['pieces']}  reward={total_reward:.0f}")

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=5)
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()
    try:
        watch(args.games, args.fps)
    except KeyboardInterrupt:
        print("\nStopped.")
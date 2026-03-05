"""
watch.py — Watch the trained agent play.

Usage:
    python3 src/watch.py
    python3 src/watch.py --games 10
    python3 src/watch.py --fps 10
"""

import os, sys, argparse, random
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import torch
from env import TetrisEnv
from model import DeepQNetwork

CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
BEST_PATH      = os.path.join(CHECKPOINT_DIR, "best.pt")
LATEST_PATH    = os.path.join(CHECKPOINT_DIR, "latest.pt")


def watch(games=5, fps=30):
    # Load model
    if os.path.exists(BEST_PATH):
        ckpt_path = BEST_PATH
    elif os.path.exists(LATEST_PATH):
        ckpt_path = LATEST_PATH
    else:
        print("No checkpoint found. Train first: python3 src/train.py")
        return

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model = DeepQNetwork().to(device)

    # Try loading as state_dict first (best.pt), then as full checkpoint (latest.pt)
    ckpt = torch.load(ckpt_path, map_location=device)
    if isinstance(ckpt, dict) and "model" in ckpt:
        model.load_state_dict(ckpt["model"])
    else:
        model.load_state_dict(ckpt)

    model.eval()
    print(f"Loaded model from {ckpt_path}")

    env = TetrisEnv(render=True, render_fps=fps)

    for g in range(1, games + 1):
        env.reset()
        done = False
        total_reward = 0

        while not done:
            next_states = env.get_next_states()
            if not next_states:
                break

            action_keys = list(next_states.keys())
            state_arrays = np.array([next_states[k][0] for k in action_keys], dtype=np.float32)
            with torch.no_grad():
                predictions = model(torch.tensor(state_arrays, device=device)).squeeze(1)
            best_idx = torch.argmax(predictions).item()
            action_key = action_keys[best_idx]

            _, reward, done, info = env.step(action_key, render=True)
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
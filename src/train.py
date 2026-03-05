"""
train.py — DQN Training (nuno-faria exact match)

3000 episodes, epsilon linear decay stops at episode 2000.
Once the agent consistently plays long games (500+ pieces),
it has effectively learned to play indefinitely.

Usage:
    python3 src/train.py
    python3 src/train.py --render
    python3 src/train.py --resume
"""

import os, sys, argparse, time, datetime
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from env import TetrisEnv
from agent import DQNAgent

EPISODES       = 3000
MAX_STEPS      = None       # no limit — let the agent play as long as it can
LOG_EVERY      = 10
SAVE_LATEST    = 50
SAVE_EVERY     = 500
RENDER_EVERY   = 50
RENDER_FPS     = 30
CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
RUNS_DIR       = os.path.join(os.path.dirname(__file__), "..", "runs")
BEST_PATH      = os.path.join(CHECKPOINT_DIR, "best.pt")
LATEST_PATH    = os.path.join(CHECKPOINT_DIR, "latest.pt")


class Logger:
    def __init__(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.file = open(path, "w", buffering=1)
        self.file.write("episode,score,lines,pieces,reward,epsilon,loss\n")
        print(f"Logging to: {path}\n")

    def write(self, ep, score, lines, pieces, reward, epsilon, loss):
        self.file.write(f"{ep},{score},{lines},{pieces},{reward:.1f},{epsilon:.4f},{loss:.5f}\n")

    def close(self):
        self.file.close()


def train(render=False, resume=False):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(RUNS_DIR, exist_ok=True)

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = Logger(os.path.join(RUNS_DIR, f"train_{ts}.log"))

    env = TetrisEnv(render=False, render_fps=RENDER_FPS)
    agent = DQNAgent(
        state_size=env.get_state_size(),
        epsilon_stop_episode=2000,   # nuno-faria: epsilon=0 at ep 2000
        replay_size=1000,            # nuno-faria: small buffer
        batch_size=128,              # nuno-faria: 128
        replay_start_size=128,
        epochs=1,
        discount=0.95,
    )

    if resume and os.path.exists(LATEST_PATH):
        agent.load(LATEST_PATH)

    best_score = 0
    recent_scores = []

    print(f"Training for {EPISODES} episodes")
    print(f"Epsilon will reach 0 at episode 2000\n")

    for ep in range(1, EPISODES + 1):
        if render and ep % RENDER_EVERY == 0:
            env.render_mode = True
            env._init_render()
        else:
            env.render_mode = False

        current_state = env.reset()
        done = False
        ep_reward = 0
        steps = 0
        t0 = time.time()

        while not done and (MAX_STEPS is None or steps < MAX_STEPS):
            next_states = env.get_next_states()
            if not next_states:
                break

            best_idx, best_state = agent.best_state(next_states)

            reward, done, info = env.step(best_idx)
            ep_reward += reward
            steps += 1

            agent.add_to_memory(current_state, best_state, reward, done)
            current_state = best_state

        loss = agent.train()
        agent.decay_epsilon()

        score = info["score"]
        lines = info["lines"]
        pieces = info.get("pieces", 0)
        loss_val = loss if loss is not None else 0.0

        recent_scores.append(score)
        if len(recent_scores) > 50:
            recent_scores.pop(0)

        if score > best_score:
            best_score = score
            agent.save(BEST_PATH)

        if ep % SAVE_EVERY == 0:
            agent.save(os.path.join(CHECKPOINT_DIR, f"checkpoint_ep{ep}.pt"))
        if ep % SAVE_LATEST == 0:
            agent.save(LATEST_PATH)

        logger.write(ep, score, lines, pieces, ep_reward, agent.epsilon, loss_val)

        if ep % LOG_EVERY == 0:
            dt = time.time() - t0
            bar_len = 20
            filled = int(bar_len * ep / EPISODES)
            bar = "█" * filled + "░" * (bar_len - filled)
            pct = 100 * ep / EPISODES
            avg50 = np.mean(recent_scores) if recent_scores else 0
            print(
                f"[{bar}] {pct:5.1f}%  "
                f"ep={ep:>5}  score={score:>7}  best={best_score:>7}  "
                f"avg50={avg50:>7.0f}  lines={lines:>4}  pieces={pieces:>4}  "
                f"ε={agent.epsilon:.3f}  loss={loss_val:.4f}  t={dt:.1f}s"
            )

    agent.save(LATEST_PATH)
    print(f"\nDone. Best score: {best_score}")
    logger.close()
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
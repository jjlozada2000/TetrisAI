
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from agent import SearchAgent, SearchConfig
from env import TetrisEnv
from model import load_weights

def watch(games, fps, depth, take, max_pieces):
    weights = load_weights()
    print("Using weights: built-in defaults" if not os.path.exists(os.path.join(os.path.dirname(__file__), "..", "models", "heuristic_weights.json")) else "Using weights: saved heuristic file")
    print(f"Search depth={depth}, beam={take}, piece cap={max_pieces}")

    env = TetrisEnv(render=True, render_fps=fps)
    agent = SearchAgent(weights=weights, config=SearchConfig(depth=depth, take=take))

    for g in range(1, games + 1):
        env.reset()
        done = False
        while not done and env.stats.pieces < max_pieces:
            action = agent.choose_action(env)
            _, _, done, info = env.step(action, render=True)
        info = env._info()
        print(
            f"Game {g}/{games} | score={info['score']} lines={info['lines']} pieces={info['pieces']} "
            f"| nodes={info.get('nodes', 0)} cache_hits={info.get('cache_hits', 0)}"
        )
    env.close()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=1)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--take", type=int, default=8)
    ap.add_argument("--max-pieces", type=int, default=5000)
    args = ap.parse_args()
    watch(args.games, args.fps, args.depth, args.take, args.max_pieces)

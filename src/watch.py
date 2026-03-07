import argparse
import json
import os

from agent import Agent
from env import TetrisEnv
from model import BOT_PRESETS

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")


def load_weights(bot):
    path = os.path.join(MODELS_DIR, f"{bot}_weights.json")
    if os.path.exists(path):
        with open(path, "r") as f:
            print(f"Using weights: {path}")
            return json.load(f)
    print("Using weights: built-in defaults")
    return dict(BOT_PRESETS[bot])


def watch(bot="tree", games=1, fps=30, depth=2, beam=8, max_pieces=5000):
    weights = load_weights(bot)
    env = TetrisEnv(render=True, render_fps=fps, max_pieces=max_pieces)
    agent = Agent(bot=bot, weights=weights, depth=(1 if bot == "greedy" else depth), beam=beam)

    print(f"Search depth={1 if bot == 'greedy' else depth}, beam={beam}, piece cap={max_pieces}")

    for g in range(1, games + 1):
        env.reset()
        done = False
        info = {}
        while not done:
            action = agent.choose_action(env)
            _, _, done, info = env.step(action, render=True)
            env._render(bot_name=bot)

        print(
            f"Game {g}/{games}  score={info['score']}  "
            f"lines={info['lines']}  pieces={info['pieces']}  "
            f"nodes={info.get('nodes', 0)}  cache_hits={info.get('cache_hits', 0)}"
        )

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bot", choices=["greedy", "tree", "perfect_clear"], default="tree")
    parser.add_argument("--games", type=int, default=1)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--take", type=int, default=8)
    parser.add_argument("--max-pieces", type=int, default=5000)
    args = parser.parse_args()

    try:
        watch(args.bot, args.games, args.fps, args.depth, args.take, args.max_pieces)
    except KeyboardInterrupt:
        print("\nStopped.")
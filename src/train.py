import argparse
import json
import os
import random

from agent import Agent
from env import TetrisEnv
from model import BOT_PRESETS

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)


def weights_path(bot):
    return os.path.join(MODELS_DIR, f"{bot}_weights.json")


def load_weights(bot):
    path = weights_path(bot)
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return dict(BOT_PRESETS[bot])


def save_weights(bot, weights):
    path = weights_path(bot)
    with open(path, "w") as f:
        json.dump(weights, f, indent=2)
    print(f"Saved → {path}", flush=True)


def mutate(weights, scale=0.20):
    out = dict(weights)
    for k in out:
        out[k] += random.uniform(-scale, scale)
    return out


def evaluate(bot, weights, episodes, depth, beam, max_pieces):
    total_score = 0.0
    total_lines = 0.0
    total_pieces = 0.0

    for _ in range(episodes):
        env = TetrisEnv(render=False, max_pieces=max_pieces)
        agent = Agent(bot=bot, weights=weights, depth=depth, beam=beam)

        done = False
        info = {"score": 0, "lines": 0, "pieces": 0}
        while not done:
            action = agent.choose_action(env)
            _, _, done, info = env.step(action)

        total_score += info["score"]
        total_lines += info["lines"]
        total_pieces += info["pieces"]

    return {
        "score": total_score / episodes,
        "lines": total_lines / episodes,
        "pieces": total_pieces / episodes,
    }


def train_one(bot, iterations, episodes, depth, beam, max_pieces, scale):
    print(f"\nTraining bot: {bot}", flush=True)

    best = load_weights(bot)
    best_eval = evaluate(bot, best, episodes, depth, beam, max_pieces)
    print(
        f"Initial  score={best_eval['score']:.1f}  "
        f"lines={best_eval['lines']:.1f}  pieces={best_eval['pieces']:.1f}",
        flush=True,
    )

    for i in range(iterations):
        cand = mutate(best, scale=scale)
        ev = evaluate(bot, cand, episodes, depth, beam, max_pieces)

        print(
            f"[{bot}] iter {i+1}/{iterations}  "
            f"score={ev['score']:.1f}  lines={ev['lines']:.1f}  pieces={ev['pieces']:.1f}",
            flush=True,
        )

        if ev["score"] > best_eval["score"]:
            best = cand
            best_eval = ev
            save_weights(bot, best)
            print(f"[{bot}] new best!", flush=True)

    save_weights(bot, best)
    print(f"[{bot}] done.\n", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bot", choices=["greedy", "tree", "perfect_clear"])
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--take", type=int, default=8)
    parser.add_argument("--max-pieces", type=int, default=400)
    parser.add_argument("--scale", type=float, default=0.20)
    args = parser.parse_args()

    if args.all:
        for bot_name in ["greedy", "tree", "perfect_clear"]:
            bot_depth = 1 if bot_name == "greedy" else args.depth
            train_one(
                bot_name,
                args.iterations,
                args.episodes,
                bot_depth,
                args.take,
                args.max_pieces,
                args.scale,
            )
    else:
        bot_name = args.bot or "tree"
        bot_depth = 1 if bot_name == "greedy" else args.depth
        train_one(
            bot_name,
            args.iterations,
            args.episodes,
            bot_depth,
            args.take,
            args.max_pieces,
            args.scale,
        )
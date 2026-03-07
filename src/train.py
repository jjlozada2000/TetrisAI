
import argparse
import math
import os
import random
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))

from agent import SearchAgent, SearchConfig
from env import TetrisEnv
from model import load_weights, save_weights

TUNE_KEYS = [
    "aggregate_height", "holes", "bumpiness", "completed_lines", "max_height",
    "row_transitions", "col_transitions", "covered_cells", "right_well_depth",
    "right_well_open", "right_well_blocked", "well_lock_bonus", "well_lock_break_penalty",
    "i_ready_bonus", "perfect_clear_bonus", "perfect_clear_setup", "tetris_bonus", "danger_penalty",
]

def evaluate(weights, episodes, depth, take, max_pieces, seed):
    total_scores, total_lines, total_pieces = [], [], []
    for ep in range(episodes):
        env = TetrisEnv(render=False, seed=seed + ep)
        env.reset()
        agent = SearchAgent(weights=weights, config=SearchConfig(depth=depth, take=take))
        done = False
        while not done and env.stats.pieces < max_pieces:
            action = agent.choose_action(env)
            _, _, done, _ = env.step(action, render=False)
        total_scores.append(env.stats.score)
        total_lines.append(env.stats.lines)
        total_pieces.append(env.stats.pieces)
        env.close()
    return (
        statistics.mean(total_scores),
        statistics.mean(total_lines),
        statistics.mean(total_pieces),
    )

def mutate(weights, scale, rng):
    child = dict(weights)
    for k in TUNE_KEYS:
        sigma = max(0.05, abs(child[k]) * scale)
        child[k] += rng.gauss(0.0, sigma)
    return child

def train(iterations, episodes, depth, take, max_pieces, seed, scale):
    rng = random.Random(seed)
    best = load_weights()
    best_score, best_lines, best_pieces = evaluate(best, episodes, depth, take, max_pieces, seed)
    print(f"Starting search-weight tuning", flush=True)
    print(f"Base score={best_score:.0f} lines={best_lines:.1f} pieces={best_pieces:.1f}", flush=True)

    for i in range(1, iterations + 1):
        t0 = time.time()
        candidate = mutate(best, scale, rng)
        score, lines, pieces = evaluate(candidate, episodes, depth, take, max_pieces, seed + 1000 * i)
        improved = False

        # favor score first, then lines
        if score > best_score or (math.isclose(score, best_score) and lines > best_lines):
            best = candidate
            best_score, best_lines, best_pieces = score, lines, pieces
            save_weights(best)
            improved = True

        marker = "*" if improved else " "
        print(
            f"{marker} iter {i:>4}/{iterations} | cand_score={score:>7.0f} cand_lines={lines:>5.1f} "
            f"| best_score={best_score:>7.0f} best_lines={best_lines:>5.1f} best_pieces={best_pieces:>5.1f} "
            f"| dt={time.time()-t0:.1f}s",
            flush=True,
        )

    save_weights(best)
    print("\nTraining complete. Saved best heuristic weights.", flush=True)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--iterations", type=int, default=30)
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--take", type=int, default=8)
    ap.add_argument("--max-pieces", type=int, default=400)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--scale", type=float, default=0.12)
    args = ap.parse_args()
    train(args.iterations, args.episodes, args.depth, args.take, args.max_pieces, args.seed, args.scale)

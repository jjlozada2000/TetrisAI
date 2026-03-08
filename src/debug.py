"""
debug.py — Quick sanity checks for the search bot.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from agent import TetrisSearchAgent
from env import TetrisEnv


if __name__ == "__main__":
    env = TetrisEnv(render=False)
    env.reset(seed=0)
    agent = TetrisSearchAgent(depth=2, take=8)

    print("Initial legal actions:", len(env.get_next_states()))
    for step in range(20):
        action = agent.choose_action(env)
        _, reward, done, info = env.step(action)
        print(
            f"step={step+1:>2} action={action} reward={reward:>8.2f} "
            f"score={info['score']:>5} lines={info['lines']:>3} pieces={info['pieces']:>3}"
        )
        if done:
            print("Game ended during sanity check")
            break
    env.close()

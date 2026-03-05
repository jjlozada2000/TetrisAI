# TetrisAI
First attempt to try to learn how AI works in game spaces, more specifically for the game of Tetris.
# Tetris RL

A Tetris implementation in Pygame, built as the foundation for a Deep Q-Network (DQN) reinforcement learning agent.

## Setup

```bash
pip install -r requirements.txt
python game.py
```

## Controls

| Key | Action |
|-----|--------|
| ← → | Move left / right |
| ↑ | Rotate |
| ↓ | Soft drop |
| Space | Hard drop |
| P | Pause |
| R | Restart |

## Project structure

```
tetris/
├── game.py           # Pygame Tetris game + live stats panel
├── requirements.txt
└── README.md
```

## Roadmap

- [x] Playable Tetris with ghost piece
- [x] Live score / lines / level / time panel
- [x] Score and lines history graphs
- [ ] Environment wrapper (for RL agent)
- [ ] DQN agent (model + training loop)
- [ ] Reward function tuning
- [ ] Save / load trained model
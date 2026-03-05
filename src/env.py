"""
env.py — Tetris RL Environment Wrapper

Exposes a Gymnasium-style interface:
    obs, info = env.reset()
    obs, reward, done, info = env.step(action)

Actions (6):
    0 = move left
    1 = move right
    2 = rotate
    3 = soft drop
    4 = hard drop
    5 = do nothing (1 natural fall tick)

State (observation) — 1D numpy array of floats:
    - Board columns heights          (10 values)
    - Holes per column               (10 values)
    - Bumpiness per adjacent pair    (9 values)
    - Max height                     (1 value)
    - Lines cleared this game        (1 value)
    - Current piece type one-hot     (7 values)
    - Next piece type one-hot        (7 values)
    Total: 45 values
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pygame
from game import (
    Board, Piece, Stats,
    COLS, ROWS, CELL, BOARD_W, BOARD_H, PANEL_W, WINDOW_W, WINDOW_H,
    BG, FPS,
    draw_board, draw_piece, draw_panel,
)

# ── Action space ──────────────────────────────────────────────────────────────

ACTIONS = {
    0: "left",
    1: "right",
    2: "rotate",
    3: "soft_drop",
    4: "hard_drop",
    5: "nothing",
}
N_ACTIONS = len(ACTIONS)

# ── Piece index map (for one-hot encoding) ────────────────────────────────────

PIECE_KINDS = ["I", "O", "T", "S", "Z", "J", "L"]
PIECE_IDX   = {k: i for i, k in enumerate(PIECE_KINDS)}

# ── Reward constants ──────────────────────────────────────────────────────────

R_SURVIVAL          =  0.01   # per piece placed — rewards staying alive
R_LINE_1            =  1.0
R_LINE_2            =  3.0
R_LINE_3            =  5.0
R_TETRIS            = 12.0    # 4-line clear with I-piece
R_TETRIS_BONUS      =  8.0    # extra on top for I-piece specifically
R_PERFECT_CLEAR     = 30.0    # full board clear
R_TSPIN_SINGLE      =  4.0
R_TSPIN_DOUBLE      = 10.0
R_TSPIN_TRIPLE      = 20.0    # rare but possible
R_TWIST_BONUS       =  2.0    # non-T piece twist/kick bonus
R_HARD_DROP_CELL    =  0.02   # per cell dropped via hard drop (rewards decisiveness)
R_HOLE_PENALTY      = -0.5    # per hole created
R_HEIGHT_PENALTY    = -0.2    # per row above danger threshold
R_DANGER_THRESHOLD  = ROWS - 5  # 15 rows — start penalizing here
R_DEATH             = -20.0

LINE_REWARDS = {1: R_LINE_1, 2: R_LINE_2, 3: R_LINE_3, 4: R_TETRIS}


# ── Helpers ───────────────────────────────────────────────────────────────────

def board_heights(grid):
    """Height of each column (0 = empty col)."""
    heights = []
    for c in range(COLS):
        h = 0
        for r in range(ROWS):
            if grid[r][c]:
                h = ROWS - r
                break
        heights.append(h)
    return heights


def board_holes(grid, heights):
    """Number of empty cells beneath the top filled cell, per column."""
    holes = []
    for c in range(COLS):
        top = ROWS - heights[c]
        count = sum(1 for r in range(top + 1, ROWS) if not grid[r][c])
        holes.append(count)
    return holes


def bumpiness(heights):
    """Absolute height differences between adjacent columns."""
    return [abs(heights[i] - heights[i + 1]) for i in range(len(heights) - 1)]


def one_hot(kind):
    v = [0.0] * len(PIECE_KINDS)
    v[PIECE_IDX[kind]] = 1.0
    return v


def detect_tspin(board, piece, rotated):
    """
    Simple T-spin detection: T-piece was just rotated and at least 3 of the
    4 corner cells around the T centre are occupied (wall or filled cell).
    Returns True if a T-spin condition is met.
    """
    if piece.kind != "T":
        return False
    # Centre of the T after rotation
    cx = piece.x + 1
    cy = piece.y + 1
    corners = [(cx - 1, cy - 1), (cx + 1, cy - 1),
               (cx - 1, cy + 1), (cx + 1, cy + 1)]
    filled = 0
    for bx, by in corners:
        if bx < 0 or bx >= COLS or by < 0 or by >= ROWS:
            filled += 1  # wall counts as filled
        elif by >= 0 and board.grid[by][bx]:
            filled += 1
    return filled >= 3


# ── Environment ───────────────────────────────────────────────────────────────

class TetrisEnv:
    """
    Tetris RL environment.

    Parameters
    ----------
    render : bool
        If True, opens a Pygame window and renders each step.
        Set False for fast headless training.
    render_fps : int
        Target FPS when rendering (lower = slower/watchable, higher = fast).
    """

    def __init__(self, render: bool = False, render_fps: int = FPS):
        self.render_mode = render
        self.render_fps  = render_fps
        self._screen     = None
        self._clock      = None
        self._fonts      = None

        # These are set on reset()
        self.board  = None
        self.stats  = None
        self.piece  = None
        self.next   = None
        self.done   = False
        self._last_tspin  = False   # was the last rotation a T-spin?
        self._land_steps  = 0       # steps spent sitting on ground
        self._LOCK_STEPS  = 30      # ~0.5s at 60fps — standard Tetris lock delay

    # ── Public API ────────────────────────────────────────────────────────────

    def reset(self):
        """Start a new game. Returns initial observation."""
        self.board = Board()
        self.stats = Stats()
        self.piece = Piece()
        self.next  = Piece()
        self.done  = False
        self._last_tspin = False
        self._land_steps = 0

        if self.render_mode:
            self._init_render()

        return self._observe(), {}

    def step(self, action: int):
        """
        Apply action, advance one game tick.

        Returns
        -------
        obs     : np.ndarray  — new state
        reward  : float
        done    : bool
        info    : dict        — diagnostic info (not used for training)
        """
        assert not self.done, "Call reset() before stepping after game over."

        reward = 0.0
        placed = False          # did this action place a piece?
        tspin  = False
        twist  = False

        act = ACTIONS[action]

        # ── Apply action ──────────────────────────────────────────────────────
        if act == "left":
            if self.board.valid(self.piece, ox=-1):
                self.piece.x -= 1

        elif act == "right":
            if self.board.valid(self.piece, ox=1):
                self.piece.x += 1

        elif act == "rotate":
            rotated = self.piece.rotated()
            for kick in [0, -1, 1, -2, 2]:
                if self.board.valid(self.piece, ox=kick, matrix=rotated):
                    # Check T-spin before applying
                    if self.piece.kind == "T":
                        tspin = detect_tspin(self.board, self.piece, rotated)
                    else:
                        # Reward non-T twist (wall kick used)
                        twist = (kick != 0)
                    self.piece.x     += kick
                    self.piece.matrix = rotated
                    break

        elif act == "soft_drop":
            if self.board.valid(self.piece, oy=1):
                self.piece.y += 1
            else:
                placed = True

        elif act == "hard_drop":
            cells_dropped = 0
            while self.board.valid(self.piece, oy=1):
                self.piece.y += 1
                cells_dropped += 1
            reward += cells_dropped * R_HARD_DROP_CELL
            placed = True

        elif act == "nothing":
            # Natural gravity tick
            if self.board.valid(self.piece, oy=1):
                self.piece.y += 1
                self._land_steps = 0   # reset lock timer — still falling
            else:
                placed = True

        # ── Lock delay — piece must sit on ground for _LOCK_STEPS before locking ──
        # Hard drop bypasses this (placed=True already set above).
        # Moves/rotations while grounded reset the counter (standard Tetris rule).
        if not placed:
            grounded = not self.board.valid(self.piece, oy=1)
            if grounded:
                if act in ("left", "right", "rotate"):
                    self._land_steps = 0   # player acted — reset lock timer
                else:
                    self._land_steps += 1
                if self._land_steps >= self._LOCK_STEPS:
                    placed = True
                    self._land_steps = 0
            else:
                self._land_steps = 0

        # ── Place piece and compute rewards when locked ───────────────────────
        if placed:
            holes_before = sum(board_holes(self.board.grid,
                                           board_heights(self.board.grid)))

            self.board.place(self.piece)
            cleared = self.board.clear_lines()

            holes_after  = sum(board_holes(self.board.grid,
                                            board_heights(self.board.grid)))
            heights_now  = board_heights(self.board.grid)
            max_h        = max(heights_now)

            # ── Reward: line clears ───────────────────────────────────────────
            if cleared > 0:
                reward += LINE_REWARDS.get(cleared, R_TETRIS)

                # Tetris bonus (4 lines cleared with I-piece)
                if cleared == 4 and self.piece.kind == "I":
                    reward += R_TETRIS_BONUS

                # T-spin bonuses
                if tspin:
                    tspin_bonus = {1: R_TSPIN_SINGLE,
                                   2: R_TSPIN_DOUBLE,
                                   3: R_TSPIN_TRIPLE}.get(cleared, R_TSPIN_SINGLE)
                    reward += tspin_bonus

                # Twist bonus (non-T piece used wall kick to clear)
                if twist and cleared > 0:
                    reward += R_TWIST_BONUS

                # Perfect clear
                if all(not any(row) for row in self.board.grid):
                    reward += R_PERFECT_CLEAR

            # ── Reward: survival ──────────────────────────────────────────────
            reward += R_SURVIVAL

            # ── Penalty: new holes ────────────────────────────────────────────
            new_holes = max(0, holes_after - holes_before)
            reward += new_holes * R_HOLE_PENALTY

            # ── Penalty: height approaching danger zone ───────────────────────
            if max_h > R_DANGER_THRESHOLD:
                excess = max_h - R_DANGER_THRESHOLD
                reward += excess * R_HEIGHT_PENALTY

            # ── Update stats ──────────────────────────────────────────────────
            self.stats.add_lines(cleared, self.stats.level)
            self.stats.record()

            # ── Spawn next piece ──────────────────────────────────────────────
            self.piece = self.next
            self.next  = Piece()

            if not self.board.valid(self.piece):
                reward += R_DEATH
                self.done = True

        # ── Render ────────────────────────────────────────────────────────────
        if self.render_mode:
            self._render()

        info = {
            "score":   self.stats.score,
            "lines":   self.stats.lines,
            "level":   self.stats.level,
            "tspin":   tspin,
            "twist":   twist,
        }

        return self._observe(), reward, self.done, info

    def close(self):
        if self._screen:
            pygame.quit()
            self._screen = None

    # ── Observation ───────────────────────────────────────────────────────────

    def _observe(self):
        grid    = self.board.grid
        heights = board_heights(grid)
        holes   = board_holes(grid, heights)
        bumps   = bumpiness(heights)
        max_h   = max(heights)
        cur_oh  = one_hot(self.piece.kind)
        nxt_oh  = one_hot(self.next.kind)

        obs = (
            [h / ROWS for h in heights]      +   # normalised 0-1
            [h / (ROWS * COLS) for h in holes] +
            [b / ROWS for b in bumps]         +
            [max_h / ROWS]                    +
            [self.stats.lines / 999]          +   # soft cap normalisation
            cur_oh                            +
            nxt_oh
        )
        return np.array(obs, dtype=np.float32)

    @property
    def observation_size(self):
        return 45

    # ── Rendering ─────────────────────────────────────────────────────────────

    def _init_render(self):
        if self._screen:
            return
        pygame.init()
        self._screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
        pygame.display.set_caption("Tetris RL — Training")
        self._clock = pygame.time.Clock()
        self._fonts = (
            pygame.font.SysFont("menlo", 28, bold=True),
            pygame.font.SysFont("menlo", 20),
            pygame.font.SysFont("menlo", 14),
        )

    def _render(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close(); sys.exit()

        self._screen.fill(BG)

        board_surf = pygame.Surface((BOARD_W, BOARD_H))
        board_surf.fill(BG)
        draw_board(board_surf, self.board)

        gy = self.board.ghost_y(self.piece)
        draw_piece(board_surf, self.piece, ghost_y=gy)
        self._screen.blit(board_surf, (0, 0))

        draw_panel(self._screen, self.stats, self.next, *self._fonts)

        pygame.display.flip()
        self._clock.tick(self.render_fps)


# ── Quick sanity check ────────────────────────────────────────────────────────

if __name__ == "__main__":
    import random

    print("Running random agent for 200 steps (headless)...")
    env = TetrisEnv(render=False)
    obs, _ = env.reset()

    print(f"Observation size: {len(obs)}")

    total_reward = 0.0
    for step in range(200):
        action = random.randint(0, N_ACTIONS - 1)
        obs, reward, done, info = env.step(action)
        total_reward += reward
        if done:
            print(f"  Game over at step {step} | score={info['score']} lines={info['lines']}")
            obs, _ = env.reset()
            total_reward = 0.0

    print("Sanity check passed.")
    env.close()
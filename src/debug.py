"""
env.py — Tetris RL Environment (Placement-Based Action Space)

Instead of choosing one keypress at a time, the agent chooses the FINAL
PLACEMENT of each piece — which column and rotation to use. The environment
then hard-drops the piece there instantly.

Why this works better:
    - Every action = one placed piece = immediate reward feedback
    - No sparse reward problem (agent doesn't need 30 random moves to learn)
    - Proven approach used by most successful Tetris AIs

Action space:
    For each piece there are up to 4 rotations × 10 columns = 40 possible
    placements. Invalid placements (piece doesn't fit) are masked out.
    N_ACTIONS = 40

State (observation) — 1D numpy array of 45 floats:
    - Column heights × 10          (normalised / ROWS)
    - Holes per column × 10        (normalised / ROWS*COLS)
    - Bumpiness × 9                (normalised / ROWS)
    - Max height × 1               (normalised / ROWS)
    - Lines cleared this game × 1  (normalised / 999)
    - Current piece one-hot × 7
    - Next piece one-hot × 7
    Total: 45
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import copy
import numpy as np
import pygame

from game import (
    Board, Piece, Stats,
    COLS, ROWS, CELL, BOARD_W, BOARD_H, PANEL_W, WINDOW_W, WINDOW_H,
    BG, FPS,
    draw_board, draw_piece, draw_panel,
)

# ── Action space ──────────────────────────────────────────────────────────────

N_ROTATIONS = 4
N_ACTIONS   = N_ROTATIONS * COLS   # 40

def action_to_placement(action: int):
    """Convert action index → (rotation, col)."""
    rotation = action // COLS
    col      = action  % COLS
    return rotation, col

# ── Piece index map ───────────────────────────────────────────────────────────

PIECE_KINDS = ["I", "O", "T", "S", "Z", "J", "L"]
PIECE_IDX   = {k: i for i, k in enumerate(PIECE_KINDS)}

# ── Reward constants ──────────────────────────────────────────────────────────

R_SURVIVAL         =  1.0
R_LINE_1           =  5.0
R_LINE_2           = 15.0
R_LINE_3           = 25.0
R_TETRIS           = 60.0
R_TETRIS_BONUS     = 40.0
R_PERFECT_CLEAR    = 150.0
R_HOLE_PENALTY     = -2.0
R_BUMP_PENALTY     = -0.5
R_HEIGHT_PENALTY   = -1.0
R_DANGER_THRESHOLD = ROWS - 5
R_DEATH            = -20.0

LINE_REWARDS = {1: R_LINE_1, 2: R_LINE_2, 3: R_LINE_3, 4: R_TETRIS}


# ── Helpers ───────────────────────────────────────────────────────────────────

def board_heights(grid):
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
    holes = []
    for c in range(COLS):
        top   = ROWS - heights[c]
        count = sum(1 for r in range(top + 1, ROWS) if not grid[r][c])
        holes.append(count)
    return holes


def bumpiness(heights):
    return [abs(heights[i] - heights[i + 1]) for i in range(len(heights) - 1)]


def one_hot(kind):
    v = [0.0] * len(PIECE_KINDS)
    v[PIECE_IDX[kind]] = 1.0
    return v


def rotate_matrix(matrix, times):
    m = [row[:] for row in matrix]
    for _ in range(times % 4):
        m = [list(row) for row in zip(*m[::-1])]
    return m


def get_valid_placements(board, piece):
    """
    Returns list of (rotation, col, final_row, matrix) for every
    valid placement of the piece on the current board.
    """
    placements = []
    seen = set()
    for rot in range(N_ROTATIONS):
        matrix  = rotate_matrix(piece.matrix, rot)
        piece_w = len(matrix[0])

        for col in range(COLS - piece_w + 1):
            test = copy.deepcopy(piece)
            test.matrix = matrix
            test.x = col
            test.y = 0

            if not board.valid(test):
                continue

            while board.valid(test, oy=1):
                test.y += 1

            key = (rot, col)
            if key not in seen:
                seen.add(key)
                placements.append((rot, col, test.y, matrix))

    return placements


# ── Environment ───────────────────────────────────────────────────────────────

class TetrisEnv:
    def __init__(self, render: bool = False, render_fps: int = FPS):
        self.render_mode = render
        self.render_fps  = render_fps
        self._screen     = None
        self._clock      = None
        self._fonts      = None

        self.board  = None
        self.stats  = None
        self.piece  = None
        self.next   = None
        self.done   = False
        self._valid_placements = []

    def reset(self):
        self.board = Board()
        self.stats = Stats()
        self.piece = Piece()
        self.next  = Piece()
        self.done  = False
        self._valid_placements = get_valid_placements(self.board, self.piece)

        if self.render_mode:
            self._init_render()

        return self._observe(), {}

    def step(self, action: int):
        assert not self.done, "Call reset() before stepping after game over."

        reward = 0.0

        placement = self._find_placement(*action_to_placement(action))

        if placement is None:
            reward += R_DEATH
            self.done = True
            return self._observe(), reward, self.done, self._info()

        rot, final_col, final_row, matrix = placement

        heights_before = board_heights(self.board.grid)
        holes_before   = sum(board_holes(self.board.grid, heights_before))
        bumps_before   = sum(bumpiness(heights_before))

        self.piece.matrix = matrix
        self.piece.x      = final_col
        self.piece.y      = final_row
        self.board.place(self.piece)
        cleared = self.board.clear_lines()

        heights_after = board_heights(self.board.grid)
        holes_after   = sum(board_holes(self.board.grid, heights_after))
        bumps_after   = sum(bumpiness(heights_after))
        max_h         = max(heights_after)

        reward += R_SURVIVAL

        if cleared > 0:
            reward += LINE_REWARDS.get(cleared, R_TETRIS)
            if cleared == 4 and self.piece.kind == "I":
                reward += R_TETRIS_BONUS
            if all(not any(row) for row in self.board.grid):
                reward += R_PERFECT_CLEAR

        new_holes = max(0, holes_after - holes_before)
        reward += new_holes * R_HOLE_PENALTY

        bump_increase = max(0, bumps_after - bumps_before)
        reward += bump_increase * R_BUMP_PENALTY

        if max_h > R_DANGER_THRESHOLD:
            excess = max_h - R_DANGER_THRESHOLD
            reward += excess * R_HEIGHT_PENALTY

        self.stats.add_lines(cleared, self.stats.level)
        self.stats.record()

        self.piece = self.next
        self.next  = Piece()
        self._valid_placements = get_valid_placements(self.board, self.piece)

        if not self._valid_placements:
            reward += R_DEATH
            self.done = True

        if self.render_mode:
            self._render()

        return self._observe(), reward, self.done, self._info()

    def get_valid_action_mask(self):
        mask = np.zeros(N_ACTIONS, dtype=bool)
        for rot, col, _, _ in self._valid_placements:
            action = rot * COLS + col
            if action < N_ACTIONS:
                mask[action] = True
        return mask

    def close(self):
        if self._screen:
            pygame.quit()
            self._screen = None

    def _observe(self):
        grid    = self.board.grid
        heights = board_heights(grid)
        holes   = board_holes(grid, heights)
        bumps   = bumpiness(heights)
        max_h   = max(heights)
        cur_oh  = one_hot(self.piece.kind)
        nxt_oh  = one_hot(self.next.kind)

        obs = (
            [h / ROWS for h in heights]          +
            [h / (ROWS * COLS) for h in holes]   +
            [b / ROWS for b in bumps]             +
            [max_h / ROWS]                        +
            [self.stats.lines / 999]              +
            cur_oh                                +
            nxt_oh
        )
        return np.array(obs, dtype=np.float32)

    @property
    def observation_size(self):
        return 45

    def _info(self):
        return {
            "score":  self.stats.score,
            "lines":  self.stats.lines,
            "level":  self.stats.level,
            "pieces": self.stats.pieces,
        }

    def _find_placement(self, rotation, col):
        if not self._valid_placements:
            return None
        for p in self._valid_placements:
            if p[0] == rotation and p[1] == col:
                return p
        same_rot = [p for p in self._valid_placements if p[0] == rotation]
        if same_rot:
            return min(same_rot, key=lambda p: abs(p[1] - col))
        return self._valid_placements[0]

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
                self.close()
                import sys; sys.exit()
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


# ── Sanity check ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import random
    print("Running placement-based env sanity check...")

    env = TetrisEnv(render=False)
    obs, _ = env.reset()
    print(f"Observation size : {len(obs)}")
    print(f"Action space     : {N_ACTIONS}")

    total_reward = 0.0
    for step in range(500):
        mask   = env.get_valid_action_mask()
        valid  = np.where(mask)[0]
        action = int(np.random.choice(valid))
        obs, reward, done, info = env.step(action)
        total_reward += reward
        if done:
            print(f"  Game over at step {step} | score={info['score']} lines={info['lines']} reward={total_reward:.1f}")
            obs, _ = env.reset()
            total_reward = 0.0

    print("Sanity check passed.")
    env.close()
"""
env.py — Tetris Environment (nuno-faria style)

Based on the proven approach from github.com/nuno-faria/tetris-ai (800K+ points).

Key design decisions:
  - State = only 4 features: lines_cleared, holes, bumpiness, total_height
  - get_next_states() returns {action: state} for ALL valid placements
  - Reward = 1 per piece placed + lines_cleared^2 * board_width
  - The agent scores each resulting state and picks the best one
"""

import sys, os
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

PIECE_KINDS = ["I", "O", "T", "S", "Z", "J", "L"]


# ── Helpers ───────────────────────────────────────────────────────────────────

def rotate_matrix(matrix, times):
    m = [row[:] for row in matrix]
    for _ in range(times % 4):
        m = [list(row) for row in zip(*m[::-1])]
    return m


def copy_grid(grid):
    return [row[:] for row in grid]


def get_col_heights(grid):
    heights = []
    for c in range(COLS):
        h = 0
        for r in range(ROWS):
            if grid[r][c]:
                h = ROWS - r
                break
        heights.append(h)
    return heights


def count_holes(grid):
    holes = 0
    for c in range(COLS):
        found_block = False
        for r in range(ROWS):
            if grid[r][c]:
                found_block = True
            elif found_block:
                holes += 1
    return holes


def get_bumpiness(heights):
    total = 0
    for i in range(len(heights) - 1):
        total += abs(heights[i] - heights[i + 1])
    return total


def get_state_props(grid):
    """
    Extract the 4 features that matter (proven by nuno-faria):
      - lines_cleared (will be added by caller)
      - holes
      - bumpiness
      - total_height
    Returns [holes, bumpiness, total_height] — caller adds lines_cleared.
    """
    heights = get_col_heights(grid)
    return [
        count_holes(grid),
        get_bumpiness(heights),
        sum(heights),
    ]


def get_valid_placements(board, piece):
    """Get all valid (rotation, col) placements."""
    placements = []
    seen = set()
    for rot in range(4):
        matrix = rotate_matrix(piece.matrix, rot)
        pw = len(matrix[0])
        for col in range(COLS - pw + 1):
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
    """
    Tetris environment following nuno-faria's proven architecture.

    Flow per step:
      1. Call get_next_states() → dict of {index: [lines, holes, bumps, height]}
      2. Agent picks the best index
      3. Call step(index) → reward, done, info
    """

    def __init__(self, render=False, render_fps=FPS):
        self.render_mode = render
        self.render_fps  = render_fps
        self._screen = None
        self._clock  = None
        self._fonts  = None
        self.board = None
        self.stats = None
        self.piece = None
        self.next  = None
        self.done  = False
        self._placements = []

    def reset(self):
        self.board = Board()
        self.stats = Stats()
        self.piece = Piece()
        self.next  = Piece()
        self.done  = False
        self._placements = get_valid_placements(self.board, self.piece)
        if self.render_mode:
            self._init_render()
        return self._get_state()

    def get_state_size(self):
        return 4  # lines_cleared, holes, bumpiness, total_height

    def _get_state(self):
        """Current board state features."""
        props = get_state_props(self.board.grid)
        return np.array([0] + props, dtype=np.float32)  # 0 lines cleared for current state

    def get_next_states(self):
        """
        For each valid placement, simulate it and return the resulting
        4-feature state. Returns list of (state_array, placement_tuple).
        """
        states = []
        for rot, col, row, matrix in self._placements:
            # Simulate placement
            grid = copy_grid(self.board.grid)
            for r, mrow in enumerate(matrix):
                for c, val in enumerate(mrow):
                    if val:
                        ny, nx = row + r, col + c
                        if 0 <= ny < ROWS and 0 <= nx < COLS:
                            grid[ny][nx] = (200, 200, 200)

            # Clear lines
            full = [i for i, grow in enumerate(grid) if all(grow)]
            for i in full:
                del grid[i]
                grid.insert(0, [None] * COLS)
            cleared = len(full)

            # Extract 4 features
            props = get_state_props(grid)
            state = np.array([cleared] + props, dtype=np.float32)
            states.append((state, (rot, col, row, matrix, cleared)))

        return states

    def step(self, placement_index):
        """Execute the placement and return (reward, done, info)."""
        assert not self.done

        if not self._placements:
            self.done = True
            return -1, True, self._info()

        idx = min(placement_index, len(self._placements) - 1)
        rot, col, row, matrix = self._placements[idx]

        # Place piece
        self.piece.matrix = matrix
        self.piece.x = col
        self.piece.y = row
        self.board.place(self.piece)
        cleared = self.board.clear_lines()

        # Reward: 1 per piece + lines^2 * width (nuno-faria's formula)
        reward = 1 + (cleared ** 2) * COLS

        self.stats.add_lines(cleared, self.stats.level)
        self.stats.record()

        # Next piece
        self.piece = self.next
        self.next = Piece()
        self._placements = get_valid_placements(self.board, self.piece)

        if not self._placements:
            self.done = True
            reward -= 1  # death penalty

        if self.render_mode:
            self._render()

        return reward, self.done, self._info()

    def _info(self):
        return {
            "score": self.stats.score,
            "lines": self.stats.lines,
            "level": self.stats.level,
            "pieces": self.stats.pieces,
        }

    def close(self):
        if self._screen:
            pygame.quit()
            self._screen = None

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
                sys.exit()
        self._screen.fill(BG)
        board_surf = pygame.Surface((BOARD_W, BOARD_H))
        board_surf.fill(BG)
        draw_board(board_surf, self.board)
        if not self.done:
            gy = self.board.ghost_y(self.piece)
            draw_piece(board_surf, self.piece, ghost_y=gy)
        self._screen.blit(board_surf, (0, 0))
        draw_panel(self._screen, self.stats, self.next, *self._fonts)
        pygame.display.flip()
        self._clock.tick(self.render_fps)


if __name__ == "__main__":
    import random
    print("Running env sanity check...")
    env = TetrisEnv()
    env.reset()
    total = 0
    for step in range(500):
        states = env.get_next_states()
        if not states:
            break
        idx = random.randint(0, len(states) - 1)
        reward, done, info = env.step(idx)
        total += reward
        if done:
            print(f"  Game over at step {step} | score={info['score']} lines={info['lines']} reward={total:.0f}")
            env.reset()
            total = 0
    print("Sanity check passed.")
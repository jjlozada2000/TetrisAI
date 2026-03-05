"""
env.py — Tetris Environment

Closely follows nuno-faria/tetris-ai and vietnh1009/Tetris-deep-Q-learning-pytorch.

State = [lines_cleared, holes, total_bumpiness, total_height]
get_next_states() returns dict: {(col, rotation): [lines, holes, bumps, height]}
Reward = 1 + (lines_cleared ** 2) * board_width
"""

import sys, os, copy, random
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pygame

from game import (
    Board, Piece, Stats, TETROMINOES, PIECE_COLORS,
    COLS, ROWS, CELL, BOARD_W, BOARD_H, PANEL_W, WINDOW_W, WINDOW_H,
    BG, FPS, draw_board, draw_piece, draw_panel,
)


def rotate_matrix(matrix, times):
    m = [row[:] for row in matrix]
    for _ in range(times % 4):
        m = [list(row) for row in zip(*m[::-1])]
    return m


def check_valid(grid, matrix, col, row):
    for r, mrow in enumerate(matrix):
        for c, val in enumerate(mrow):
            if val:
                nx, ny = col + c, row + r
                if nx < 0 or nx >= COLS or ny >= ROWS:
                    return False
                if ny >= 0 and grid[ny][nx]:
                    return False
    return True


def get_col_heights(grid):
    heights = []
    for c in range(COLS):
        for r in range(ROWS):
            if grid[r][c]:
                heights.append(ROWS - r)
                break
        else:
            heights.append(0)
    return heights


def count_holes(grid):
    holes = 0
    for c in range(COLS):
        block_found = False
        for r in range(ROWS):
            if grid[r][c]:
                block_found = True
            elif block_found:
                holes += 1
    return holes


def get_bumpiness_and_height(grid):
    heights = get_col_heights(grid)
    total_height = sum(heights)
    bumpiness = sum(abs(heights[i] - heights[i+1]) for i in range(len(heights)-1))
    return bumpiness, total_height


class TetrisEnv:
    def __init__(self, render=False, render_fps=FPS):
        self.render_mode = render
        self.render_fps = render_fps
        self._screen = None
        self._clock = None
        self._fonts = None
        self.board = None
        self.stats = None
        self.piece = None
        self.next = None
        self.done = False

    @staticmethod
    def get_state_size():
        return 4

    def reset(self):
        self.board = Board()
        self.stats = Stats()
        self.piece = Piece()
        self.next = Piece()
        self.done = False
        if self.render_mode:
            self._init_render()
        return self._get_board_props(self.board.grid, 0)

    def _get_board_props(self, grid, lines_cleared):
        holes = count_holes(grid)
        bumpiness, total_height = get_bumpiness_and_height(grid)
        return np.array([lines_cleared, holes, bumpiness, total_height], dtype=np.float32)

    def get_next_states(self):
        """
        Returns dict: {(col, rotation): state_properties}
        where state_properties = [lines_cleared, holes, bumpiness, total_height]
        """
        states = {}
        piece_matrix = self.piece.matrix

        for rot in range(4):
            rotated = rotate_matrix(piece_matrix, rot)
            pw = len(rotated[0])

            for col in range(COLS - pw + 1):
                # Check if piece fits at top
                if not check_valid(self.board.grid, rotated, col, 0):
                    continue

                # Drop to bottom
                row = 0
                while check_valid(self.board.grid, rotated, col, row + 1):
                    row += 1

                # Simulate placement
                grid = [r[:] for r in self.board.grid]
                for r, mrow in enumerate(rotated):
                    for c, val in enumerate(mrow):
                        if val:
                            ny, nx = row + r, col + c
                            if 0 <= ny < ROWS and 0 <= nx < COLS:
                                grid[ny][nx] = (200, 200, 200)

                # Clear lines
                lines_cleared = 0
                r = ROWS - 1
                while r >= 0:
                    if all(grid[r]):
                        del grid[r]
                        grid.insert(0, [None] * COLS)
                        lines_cleared += 1
                    else:
                        r -= 1

                state = self._get_board_props(grid, lines_cleared)
                states[(col, rot)] = (state, rotated, col, row, lines_cleared)

        return states

    def step(self, action_key, render=False):
        """
        Execute a placement. action_key is (col, rotation) from get_next_states().
        Returns (state_properties, reward, done, info).
        """
        next_states = self.get_next_states()

        if action_key not in next_states:
            # Fallback: pick a random valid action
            if next_states:
                action_key = random.choice(list(next_states.keys()))
            else:
                self.done = True
                return self._get_board_props(self.board.grid, 0), -1, True, self._info()

        state, rotated, col, row, lines_cleared = next_states[action_key]

        # Place piece on actual board
        self.piece.matrix = rotated
        self.piece.x = col
        self.piece.y = row
        self.board.place(self.piece)
        self.board.clear_lines()

        # Reward: nuno-faria formula
        reward = 1 + (lines_cleared ** 2) * COLS

        self.stats.add_lines(lines_cleared, self.stats.level)
        self.stats.record()

        # Next piece
        self.piece = self.next
        self.next = Piece()

        # Check game over
        if not self.get_next_states():
            self.done = True
            reward -= 1

        if self.render_mode or render:
            self._render()

        return state, reward, self.done, self._info()

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
        pygame.display.set_caption("Tetris RL")
        self._clock = pygame.time.Clock()
        self._fonts = (
            pygame.font.SysFont("menlo", 28, bold=True),
            pygame.font.SysFont("menlo", 20),
            pygame.font.SysFont("menlo", 14),
        )

    def _render(self):
        if not self._screen:
            self._init_render()
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
    print("Env sanity check...")
    env = TetrisEnv()
    state = env.reset()
    print(f"State size: {len(state)}, state: {state}")
    total = 0
    for i in range(500):
        ns = env.get_next_states()
        if not ns:
            break
        key = random.choice(list(ns.keys()))
        state, reward, done, info = env.step(key)
        total += reward
        if done:
            print(f"  Game over step {i}: score={info['score']} lines={info['lines']} reward={total:.0f}")
            state = env.reset()
            total = 0
    print("OK")
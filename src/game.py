
import sys
import time
import random
from collections import deque

import pygame

COLS = 10
ROWS = 20
CELL = 32
BOARD_W = COLS * CELL
BOARD_H = ROWS * CELL
PANEL_W = 220
WINDOW_W = BOARD_W + PANEL_W
WINDOW_H = BOARD_H

FPS = 60
BG = (20, 22, 28)
GRID_LINE = (40, 44, 54)
TEXT = (235, 235, 235)
SUBTEXT = (170, 175, 185)
GHOST_COLOR = (180, 180, 180)
PANEL_BG = (27, 30, 37)

PIECE_COLORS = {
    "I": (0, 240, 240),
    "O": (240, 240, 0),
    "T": (180, 0, 220),
    "S": (0, 220, 70),
    "Z": (220, 40, 40),
    "J": (40, 80, 240),
    "L": (240, 140, 0),
}

TETROMINOES = {
    "I": [[1, 1, 1, 1]],
    "O": [[1, 1],
          [1, 1]],
    "T": [[0, 1, 0],
          [1, 1, 1]],
    "S": [[0, 1, 1],
          [1, 1, 0]],
    "Z": [[1, 1, 0],
          [0, 1, 1]],
    "J": [[1, 0, 0],
          [1, 1, 1]],
    "L": [[0, 0, 1],
          [1, 1, 1]],
}

LINE_SCORES = {1: 100, 2: 300, 3: 500, 4: 800}

class Piece:
    def __init__(self, kind=None):
        self.kind = kind or random.choice(list(TETROMINOES))
        self.color = PIECE_COLORS[self.kind]
        self.matrix = [row[:] for row in TETROMINOES[self.kind]]
        self.x = COLS // 2 - len(self.matrix[0]) // 2
        self.y = 0

    def rotated(self):
        return [list(row) for row in zip(*self.matrix[::-1])]

class Board:
    def __init__(self):
        self.grid = [[None] * COLS for _ in range(ROWS)]

    def valid(self, piece, ox=0, oy=0, matrix=None):
        m = matrix or piece.matrix
        for r, row in enumerate(m):
            for c, val in enumerate(row):
                if not val:
                    continue
                nx, ny = piece.x + c + ox, piece.y + r + oy
                if nx < 0 or nx >= COLS or ny >= ROWS:
                    return False
                if ny >= 0 and self.grid[ny][nx]:
                    return False
        return True

    def place(self, piece):
        for r, row in enumerate(piece.matrix):
            for c, val in enumerate(row):
                if val:
                    ny, nx = piece.y + r, piece.x + c
                    if 0 <= ny < ROWS:
                        self.grid[ny][nx] = piece.color

    def clear_lines(self):
        full = [i for i, row in enumerate(self.grid) if all(row)]
        for i in reversed(full):
            del self.grid[i]
            self.grid.insert(0, [None] * COLS)
        return len(full)

    def ghost_y(self, piece):
        gy = piece.y
        while self.valid(piece, oy=gy - piece.y + 1):
            gy += 1
        return gy

class Stats:
    def __init__(self):
        self.score = 0
        self.level = 1
        self.lines = 0
        self.pieces = 0
        self.start = time.time()
        self.score_history = deque(maxlen=120)
        self.line_history = deque(maxlen=120)

    def add_lines(self, n, level):
        self.score += LINE_SCORES.get(n, 0) * level
        self.lines += n
        self.level = self.lines // 10 + 1

    def record(self):
        self.pieces += 1
        self.score_history.append(self.score)
        self.line_history.append(self.lines)

    def elapsed(self):
        return int(time.time() - self.start)

def _coerce_color(color):
    if isinstance(color, bool):
        return (100, 190, 255) if color else None
    if color is None:
        return None
    if isinstance(color, (list, tuple)) and len(color) >= 3:
        return tuple(int(v) for v in color[:3])
    return (100, 190, 255)

def draw_cell(surf, color, x, y, ghost=False):
    color = _coerce_color(color)
    if color is None:
        return
    rect = pygame.Rect(x * CELL, y * CELL, CELL - 1, CELL - 1)
    if ghost:
        s = pygame.Surface((CELL - 1, CELL - 1), pygame.SRCALPHA)
        s.fill((*GHOST_COLOR, 75))
        surf.blit(s, rect.topleft)
        pygame.draw.rect(surf, (*color, 130), rect, 1)
        return

    pygame.draw.rect(surf, color, rect)
    hi = tuple(min(255, c + 55) for c in color)
    lo = tuple(max(0, c - 55) for c in color)
    pygame.draw.line(surf, hi, rect.topleft, (rect.right - 1, rect.top), 1)
    pygame.draw.line(surf, hi, rect.topleft, (rect.left, rect.bottom - 1), 1)
    pygame.draw.line(surf, lo, (rect.left, rect.bottom - 1), (rect.right - 1, rect.bottom - 1), 1)
    pygame.draw.line(surf, lo, (rect.right - 1, rect.top), (rect.right - 1, rect.bottom - 1), 1)

def draw_grid(surf):
    for c in range(COLS + 1):
        pygame.draw.line(surf, GRID_LINE, (c * CELL, 0), (c * CELL, BOARD_H))
    for r in range(ROWS + 1):
        pygame.draw.line(surf, GRID_LINE, (0, r * CELL), (BOARD_W, r * CELL))

def draw_board(surf, board):
    draw_grid(surf)
    for r, row in enumerate(board.grid):
        for c, color in enumerate(row):
            if color:
                draw_cell(surf, color, c, r)

def draw_piece(surf, piece, ghost_y=None):
    if ghost_y is not None:
        for r, row in enumerate(piece.matrix):
            for c, val in enumerate(row):
                if val:
                    draw_cell(surf, piece.color, piece.x + c, ghost_y + r, ghost=True)
    for r, row in enumerate(piece.matrix):
        for c, val in enumerate(row):
            if val:
                draw_cell(surf, piece.color, piece.x + c, piece.y + r)

def draw_mini_piece(surf, kind, x, y):
    matrix = TETROMINOES[kind]
    color = PIECE_COLORS[kind]
    size = 18
    for r, row in enumerate(matrix):
        for c, val in enumerate(row):
            if val:
                rect = pygame.Rect(x + c * size, y + r * size, size - 1, size - 1)
                pygame.draw.rect(surf, color, rect)

def draw_panel(screen, stats, next_kind, hold_kind, fonts, extra=None):
    big, med, small = fonts
    x0 = BOARD_W
    panel = pygame.Rect(x0, 0, PANEL_W, WINDOW_H)
    pygame.draw.rect(screen, PANEL_BG, panel)

    def txt(s, x, y, font=med, color=TEXT):
        screen.blit(font.render(str(s), True, color), (x, y))

    txt("TETRIS BOT", x0 + 18, 18, big)
    txt(f"Score: {stats.score}", x0 + 18, 68)
    txt(f"Lines: {stats.lines}", x0 + 18, 98)
    txt(f"Level: {stats.level}", x0 + 18, 128)
    txt(f"Pieces: {stats.pieces}", x0 + 18, 158)
    txt(f"Time: {stats.elapsed()}s", x0 + 18, 188)

    txt("Next", x0 + 18, 240, med)
    if next_kind:
        draw_mini_piece(screen, next_kind, x0 + 24, 276)

    txt("Hold", x0 + 18, 350, med)
    if hold_kind:
        draw_mini_piece(screen, hold_kind, x0 + 24, 386)

    if extra:
        y = 470
        txt("Search", x0 + 18, y, med)
        y += 34
        for k, v in extra.items():
            txt(f"{k}: {v}", x0 + 18, y, small, SUBTEXT)
            y += 22

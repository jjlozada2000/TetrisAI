import random
import time
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

BG = (18, 18, 24)
GRID_LINE = (35, 35, 45)
TEXT = (235, 235, 235)
SUBTEXT = (170, 170, 180)
GHOST_COLOR = (180, 180, 180)

PIECE_COLORS = {
    "I": (0, 240, 240),
    "O": (240, 240, 0),
    "T": (160, 0, 240),
    "S": (0, 240, 0),
    "Z": (240, 0, 0),
    "J": (0, 0, 240),
    "L": (240, 160, 0),
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


def rotate_matrix(matrix):
    return [list(row) for row in zip(*matrix[::-1])]


class Piece:
    def __init__(self, kind=None):
        self.kind = kind or random.choice(list(TETROMINOES))
        self.color = PIECE_COLORS[self.kind]
        self.matrix = [row[:] for row in TETROMINOES[self.kind]]
        self.x = COLS // 2 - len(self.matrix[0]) // 2
        self.y = 0

    def rotated(self):
        return rotate_matrix(self.matrix)

    def clone(self):
        p = Piece(self.kind)
        p.matrix = [row[:] for row in self.matrix]
        p.x = self.x
        p.y = self.y
        return p


class Board:
    def __init__(self):
        self.grid = [[None] * COLS for _ in range(ROWS)]

    def clone(self):
        b = Board()
        b.grid = [row[:] for row in self.grid]
        return b

    def valid(self, piece, ox=0, oy=0, matrix=None):
        m = matrix or piece.matrix
        for r, row in enumerate(m):
            for c, val in enumerate(row):
                if not val:
                    continue
                nx = piece.x + c + ox
                ny = piece.y + r + oy
                if nx < 0 or nx >= COLS or ny >= ROWS:
                    return False
                if ny >= 0 and self.grid[ny][nx]:
                    return False
        return True

    def place(self, piece):
        for r, row in enumerate(piece.matrix):
            for c, val in enumerate(row):
                if val:
                    ny = piece.y + r
                    nx = piece.x + c
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
        while self.valid(piece, oy=(gy - piece.y + 1)):
            gy += 1
        return gy


class Stats:
    def __init__(self):
        self.score = 0
        self.level = 1
        self.lines = 0
        self.pieces = 0
        self.start = time.time()
        self.score_history = deque(maxlen=60)
        self.line_history = deque(maxlen=60)
        self._tick = 0

    def add_lines(self, n, level):
        pts = LINE_SCORES.get(n, 0) * level
        self.score += pts
        self.lines += n
        self.level = self.lines // 10 + 1

    def record(self):
        self.pieces += 1
        self._tick += 1
        self.score_history.append(self.score)
        self.line_history.append(self.lines)

    def elapsed(self):
        return int(time.time() - self.start)


def draw_cell(surf, color, x, y, ghost=False):
    rect = pygame.Rect(x * CELL, y * CELL, CELL - 1, CELL - 1)

    if isinstance(color, bool):
        if not color:
            return
        color = (80, 200, 255)

    if color is None:
        return

    if ghost:
        s = pygame.Surface((CELL - 1, CELL - 1), pygame.SRCALPHA)
        s.fill((*GHOST_COLOR, 80))
        surf.blit(s, rect.topleft)
        pygame.draw.rect(surf, (*color[:3], 120), rect, 1)
        return

    pygame.draw.rect(surf, color[:3], rect)

    pygame.draw.line(
        surf,
        tuple(min(255, c + 60) for c in color[:3]),
        rect.topleft,
        (rect.right - 1, rect.top),
        1,
    )
    pygame.draw.line(
        surf,
        tuple(min(255, c + 60) for c in color[:3]),
        rect.topleft,
        (rect.left, rect.bottom - 1),
        1,
    )
    pygame.draw.line(
        surf,
        tuple(max(0, c - 60) for c in color[:3]),
        (rect.left, rect.bottom - 1),
        (rect.right - 1, rect.bottom - 1),
        1,
    )
    pygame.draw.line(
        surf,
        tuple(max(0, c - 60) for c in color[:3]),
        (rect.right - 1, rect.top),
        (rect.right - 1, rect.bottom - 1),
        1,
    )


def draw_grid(surf):
    for c in range(COLS + 1):
        pygame.draw.line(surf, GRID_LINE, (c * CELL, 0), (c * CELL, BOARD_H))
    for r in range(ROWS + 1):
        pygame.draw.line(surf, GRID_LINE, (0, r * CELL), (BOARD_W, r * CELL))


def draw_board(surf, board):
    draw_grid(surf)
    for r, row in enumerate(board.grid):
        for c, cell in enumerate(row):
            if cell:
                color = (80, 200, 255) if isinstance(cell, bool) else cell
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


def draw_mini_piece(surf, kind, x0, y0):
    matrix = TETROMINOES[kind]
    color = PIECE_COLORS[kind]
    size = 18
    for r, row in enumerate(matrix):
        for c, val in enumerate(row):
            if val:
                rect = pygame.Rect(x0 + c * size, y0 + r * size, size - 1, size - 1)
                pygame.draw.rect(surf, color, rect)


def draw_panel(surf, stats, next_piece, hold_piece, title_font, body_font, small_font, bot_name="tree"):
    panel_x = BOARD_W + 16

    title = title_font.render("Tetris AI", True, TEXT)
    surf.blit(title, (panel_x, 20))

    bot = body_font.render(f"Bot: {bot_name}", True, SUBTEXT)
    surf.blit(bot, (panel_x, 58))

    lines = [
        f"Score:  {stats.score}",
        f"Lines:  {stats.lines}",
        f"Level:  {stats.level}",
        f"Pieces: {stats.pieces}",
        f"Time:   {stats.elapsed()}s",
    ]
    y = 100
    for line in lines:
        img = body_font.render(line, True, TEXT)
        surf.blit(img, (panel_x, y))
        y += 28

    nxt = body_font.render("Next", True, TEXT)
    surf.blit(nxt, (panel_x, 260))
    draw_mini_piece(surf, next_piece.kind, panel_x, 295)

    hld = body_font.render("Hold", True, TEXT)
    surf.blit(hld, (panel_x, 390))
    if hold_piece:
        draw_mini_piece(surf, hold_piece, panel_x, 425)

    info = small_font.render("1-3 change bot in website version", True, SUBTEXT)
    surf.blit(info, (panel_x, WINDOW_H - 30))
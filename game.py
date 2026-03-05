import pygame
import random
import sys
import time
from collections import deque

# ── Constants ────────────────────────────────────────────────────────────────

CELL = 30
COLS = 10
ROWS = 20
BOARD_W = COLS * CELL
BOARD_H = ROWS * CELL

PANEL_W = 320
WINDOW_W = BOARD_W + PANEL_W
WINDOW_H = BOARD_H

FPS = 60

# Colors (dark theme)
BG          = (13,  13,  23)
GRID_LINE   = (30,  30,  50)
PANEL_BG    = (18,  18,  32)
BORDER      = (60,  60,  100)
TEXT_DIM    = (100, 100, 140)
TEXT_BRIGHT = (220, 220, 255)
TEXT_ACCENT = (130, 180, 255)
GRAPH_LINE  = (80,  140, 255)
GRAPH_FILL  = (40,  80,  180)
GHOST_COLOR = (60,  60,  90)

PIECE_COLORS = {
    "I": (0,   220, 220),
    "O": (220, 220, 0),
    "T": (180, 0,   220),
    "S": (0,   220, 80),
    "Z": (220, 40,  40),
    "J": (40,  80,  220),
    "L": (220, 140, 0),
}

TETROMINOES = {
    "I": [[0,0,0,0],[1,1,1,1],[0,0,0,0],[0,0,0,0]],
    "O": [[1,1],[1,1]],
    "T": [[0,1,0],[1,1,1],[0,0,0]],
    "S": [[0,1,1],[1,1,0],[0,0,0]],
    "Z": [[1,1,0],[0,1,1],[0,0,0]],
    "J": [[1,0,0],[1,1,1],[0,0,0]],
    "L": [[0,0,1],[1,1,1],[0,0,0]],
}

LINE_SCORES = {1: 100, 2: 300, 3: 500, 4: 800}

# ── Piece ────────────────────────────────────────────────────────────────────

class Piece:
    def __init__(self, kind=None):
        self.kind = kind or random.choice(list(TETROMINOES))
        self.color = PIECE_COLORS[self.kind]
        self.matrix = [row[:] for row in TETROMINOES[self.kind]]
        self.x = COLS // 2 - len(self.matrix[0]) // 2
        self.y = 0

    def rotated(self):
        m = self.matrix
        return [list(row) for row in zip(*m[::-1])]


# ── Board ─────────────────────────────────────────────────────────────────────

class Board:
    def __init__(self):
        self.grid = [[None] * COLS for _ in range(ROWS)]

    def valid(self, piece, ox=0, oy=0, matrix=None):
        m = matrix or piece.matrix
        for r, row in enumerate(m):
            for c, val in enumerate(row):
                if val:
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
        for i in full:
            del self.grid[i]
            self.grid.insert(0, [None] * COLS)
        return len(full)

    def ghost_y(self, piece):
        gy = piece.y
        while self.valid(piece, oy=gy - piece.y + 1):
            gy += 1
        return gy


# ── Stats tracker ─────────────────────────────────────────────────────────────

class Stats:
    def __init__(self):
        self.score = 0
        self.level = 1
        self.lines = 0
        self.pieces = 0
        self.start = time.time()
        self.score_history = deque(maxlen=60)   # last 60 piece placements
        self.line_history  = deque(maxlen=60)
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


# ── Rendering helpers ─────────────────────────────────────────────────────────

def draw_cell(surf, color, x, y, alpha=255, ghost=False):
    rect = pygame.Rect(x * CELL, y * CELL, CELL - 1, CELL - 1)
    if ghost:
        s = pygame.Surface((CELL - 1, CELL - 1), pygame.SRCALPHA)
        s.fill((*GHOST_COLOR, 80))
        surf.blit(s, rect.topleft)
        pygame.draw.rect(surf, (*color, 120), rect, 1)
        return
    pygame.draw.rect(surf, color, rect)
    # shine
    pygame.draw.line(surf, tuple(min(255, c+60) for c in color),
                     rect.topleft, (rect.right-1, rect.top), 1)
    pygame.draw.line(surf, tuple(min(255, c+60) for c in color),
                     rect.topleft, (rect.left, rect.bottom-1), 1)
    # shadow
    pygame.draw.line(surf, tuple(max(0, c-60) for c in color),
                     (rect.left, rect.bottom-1), (rect.right-1, rect.bottom-1), 1)
    pygame.draw.line(surf, tuple(max(0, c-60) for c in color),
                     (rect.right-1, rect.top), (rect.right-1, rect.bottom-1), 1)


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
                if val and piece.y + r != ghost_y + r:
                    draw_cell(surf, piece.color, piece.x + c, ghost_y + r, ghost=True)
    for r, row in enumerate(piece.matrix):
        for c, val in enumerate(row):
            if val and 0 <= piece.y + r < ROWS:
                draw_cell(surf, piece.color, piece.x + c, piece.y + r)


def draw_mini_piece(surf, piece_kind, ox, oy, cell=22):
    m = TETROMINOES[piece_kind]
    color = PIECE_COLORS[piece_kind]
    for r, row in enumerate(m):
        for c, val in enumerate(row):
            if val:
                rect = pygame.Rect(ox + c * cell, oy + r * cell, cell - 1, cell - 1)
                pygame.draw.rect(surf, color, rect)


def draw_panel(surf, stats, next_piece, font_lg, font_md, font_sm):
    ox = BOARD_W
    pygame.draw.rect(surf, PANEL_BG, (ox, 0, PANEL_W, WINDOW_H))
    pygame.draw.line(surf, BORDER, (ox, 0), (ox, WINDOW_H), 2)

    pad = 20
    x = ox + pad
    y = pad

    def label(text, color=TEXT_DIM, font=font_sm):
        s = font.render(text, True, color)
        surf.blit(s, (x, y))
        return s.get_height() + 4

    def value(text, color=TEXT_BRIGHT, font=font_lg):
        s = font.render(text, True, color)
        surf.blit(s, (x, y))
        return s.get_height() + 6

    # Title
    title = font_lg.render("TETRIS", True, TEXT_ACCENT)
    surf.blit(title, (ox + PANEL_W // 2 - title.get_width() // 2, y))
    y += title.get_height() + 20

    # Divider
    pygame.draw.line(surf, BORDER, (x, y), (ox + PANEL_W - pad, y))
    y += 12

    # Score
    y += label("SCORE")
    y += value(f"{stats.score:,}")

    y += 8
    y += label("LEVEL")
    y += value(str(stats.level))

    y += 8
    y += label("LINES")
    y += value(str(stats.lines))

    y += 8
    elapsed = stats.elapsed()
    mins, secs = divmod(elapsed, 60)
    y += label("TIME")
    y += value(f"{mins:02d}:{secs:02d}")

    # Divider
    pygame.draw.line(surf, BORDER, (x, y), (ox + PANEL_W - pad, y))
    y += 16

    # Next piece
    y += label("NEXT")
    y += 8
    draw_mini_piece(surf, next_piece.kind, x, y)
    y += 5 * 22 + 16

    # Divider
    pygame.draw.line(surf, BORDER, (x, y), (ox + PANEL_W - pad, y))
    y += 16

    # Score graph
    y += label("SCORE HISTORY")
    y += 8
    graph_h = 80
    graph_w = PANEL_W - pad * 2
    draw_graph(surf, list(stats.score_history), x, y, graph_w, graph_h, GRAPH_LINE, GRAPH_FILL)
    y += graph_h + 16

    # Lines graph
    y += label("LINES CLEARED")
    y += 8
    draw_graph(surf, list(stats.line_history), x, y, graph_w, graph_h,
               (80, 220, 120), (40, 120, 60))


def draw_graph(surf, data, x, y, w, h, line_color, fill_color):
    pygame.draw.rect(surf, (25, 25, 40), (x, y, w, h))
    pygame.draw.rect(surf, BORDER, (x, y, w, h), 1)
    if len(data) < 2:
        return
    mn, mx = min(data), max(data)
    if mx == mn:
        mx = mn + 1
    pts = []
    for i, v in enumerate(data):
        px = x + int(i / (len(data) - 1) * (w - 2)) + 1
        py = y + h - 2 - int((v - mn) / (mx - mn) * (h - 4))
        pts.append((px, py))
    # fill
    poly = [(x + 1, y + h - 2)] + pts + [(pts[-1][0], y + h - 2)]
    s = pygame.Surface((w, h), pygame.SRCALPHA)
    rel = [(px - x, py - y) for px, py in poly]
    pygame.draw.polygon(s, (*fill_color, 80), rel)
    surf.blit(s, (x, y))
    # line
    pygame.draw.lines(surf, line_color, False, pts, 2)


# ── Game ──────────────────────────────────────────────────────────────────────

class Game:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
        pygame.display.set_caption("Tetris")
        self.clock = pygame.time.Clock()

        self.font_lg = pygame.font.SysFont("menlo", 28, bold=True)
        self.font_md = pygame.font.SysFont("menlo", 20)
        self.font_sm = pygame.font.SysFont("menlo", 14)

        self.reset()

    def reset(self):
        self.board  = Board()
        self.stats  = Stats()
        self.piece  = Piece()
        self.next   = Piece()
        self.game_over = False
        self.paused    = False
        self._fall_timer = 0
        self._lock_timer = 0
        self._lock_delay = 500   # ms before locking after landing
        self._locked     = False

    def fall_interval(self):
        # speeds up with level
        return max(80, 500 - (self.stats.level - 1) * 40)

    def spawn(self):
        self.piece = self.next
        self.next  = Piece()
        self.stats.record()
        if not self.board.valid(self.piece):
            self.game_over = True

    def hard_drop(self):
        gy = self.board.ghost_y(self.piece)
        self.stats.score += (gy - self.piece.y) * 2
        self.piece.y = gy
        self.lock_piece()

    def lock_piece(self):
        self.board.place(self.piece)
        cleared = self.board.clear_lines()
        if cleared:
            self.stats.add_lines(cleared, self.stats.level)
        self.spawn()
        self._fall_timer = 0
        self._lock_timer = 0
        self._locked = False

    def move(self, dx):
        if self.board.valid(self.piece, ox=dx):
            self.piece.x += dx

    def rotate(self):
        rotated = self.piece.rotated()
        # wall kick attempts
        for kick in [0, -1, 1, -2, 2]:
            if self.board.valid(self.piece, ox=kick, matrix=rotated):
                self.piece.x += kick
                self.piece.matrix = rotated
                return

    def update(self, dt):
        if self.game_over or self.paused:
            return

        self._fall_timer += dt

        landed = not self.board.valid(self.piece, oy=1)

        if landed:
            self._lock_timer += dt
            if self._lock_timer >= self._lock_delay:
                self.lock_piece()
        else:
            self._lock_timer = 0
            if self._fall_timer >= self.fall_interval():
                self._fall_timer = 0
                self.piece.y += 1

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit(); sys.exit()

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    self.reset(); return
                if self.game_over:
                    return
                if event.key == pygame.K_p:
                    self.paused = not self.paused; return
                if self.paused:
                    return

                if event.key == pygame.K_LEFT:   self.move(-1)
                if event.key == pygame.K_RIGHT:  self.move(1)
                if event.key == pygame.K_UP:     self.rotate()
                if event.key == pygame.K_SPACE:  self.hard_drop()
                if event.key == pygame.K_DOWN:
                    if self.board.valid(self.piece, oy=1):
                        self.piece.y += 1
                        self.stats.score += 1
                        self._fall_timer = 0

        # soft drop held
        keys = pygame.key.get_pressed()
        if keys[pygame.K_DOWN] and not self.game_over and not self.paused:
            pass  # handled via keydown repeat; could add held logic

    def draw(self):
        self.screen.fill(BG)

        # board surface
        board_surf = pygame.Surface((BOARD_W, BOARD_H))
        board_surf.fill(BG)
        draw_board(board_surf, self.board)

        # ghost + active piece
        if not self.game_over:
            gy = self.board.ghost_y(self.piece)
            draw_piece(board_surf, self.piece, ghost_y=gy)

        self.screen.blit(board_surf, (0, 0))

        # panel
        draw_panel(self.screen, self.stats, self.next,
                   self.font_lg, self.font_md, self.font_sm)

        # overlays
        if self.game_over:
            self._overlay("GAME OVER", "Press R to restart")
        elif self.paused:
            self._overlay("PAUSED", "Press P to resume")

        pygame.display.flip()

    def _overlay(self, title, subtitle):
        s = pygame.Surface((BOARD_W, BOARD_H), pygame.SRCALPHA)
        s.fill((0, 0, 0, 160))
        self.screen.blit(s, (0, 0))

        t1 = self.font_lg.render(title, True, TEXT_BRIGHT)
        t2 = self.font_sm.render(subtitle, True, TEXT_DIM)
        self.screen.blit(t1, (BOARD_W // 2 - t1.get_width() // 2, BOARD_H // 2 - 30))
        self.screen.blit(t2, (BOARD_W // 2 - t2.get_width() // 2, BOARD_H // 2 + 10))

    def run(self):
        while True:
            dt = self.clock.tick(FPS)
            self.handle_events()
            self.update(dt)
            self.draw()


if __name__ == "__main__":
    Game().run()
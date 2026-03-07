import copy
import os
import random
import sys
from dataclasses import dataclass
from functools import lru_cache

import pygame

from game import (
    BG,
    BOARD_H,
    BOARD_W,
    CELL,
    COLS,
    FPS,
    PANEL_W,
    ROWS,
    WINDOW_H,
    WINDOW_W,
    Board,
    Piece,
    Stats,
    TETROMINOES,
    draw_board,
    draw_panel,
    draw_piece,
)

PIECE_KINDS = ["I", "O", "T", "S", "Z", "J", "L"]


@lru_cache(maxsize=None)
def unique_rotations(kind):
    mats = []
    seen = set()
    m = tuple(tuple(x for x in row) for row in TETROMINOES[kind])
    cur = [list(row) for row in m]
    for _ in range(4):
        key = tuple(tuple(x for x in row) for row in cur)
        if key not in seen:
            seen.add(key)
            mats.append([list(row) for row in cur])
        cur = [list(row) for row in zip(*cur[::-1])]
    return tuple(tuple(tuple(x for x in row) for row in mat) for mat in mats)


def clone_grid(grid):
    return [row[:] for row in grid]


def valid_position(grid, matrix, col, row):
    for r, line in enumerate(matrix):
        for c, val in enumerate(line):
            if not val:
                continue
            nx = col + c
            ny = row + r
            if nx < 0 or nx >= COLS or ny >= ROWS:
                return False
            if ny >= 0 and grid[ny][nx]:
                return False
    return True


def drop_row(grid, matrix, col):
    row = 0
    if not valid_position(grid, matrix, col, row):
        return None
    while valid_position(grid, matrix, col, row + 1):
        row += 1
    return row


def place_matrix(grid, matrix, col, row, color=True):
    new_grid = clone_grid(grid)
    for r, line in enumerate(matrix):
        for c, val in enumerate(line):
            if val:
                ny = row + r
                nx = col + c
                if 0 <= ny < ROWS:
                    new_grid[ny][nx] = color
    return new_grid


def clear_lines(grid):
    full = [i for i, row in enumerate(grid) if all(row)]
    if not full:
        return grid, 0
    new_grid = [row[:] for i, row in enumerate(grid) if i not in full]
    while len(new_grid) < ROWS:
        new_grid.insert(0, [None] * COLS)
    return new_grid, len(full)


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


def hole_count(grid, heights=None):
    heights = heights or board_heights(grid)
    holes = 0
    for c in range(COLS):
        seen_block = False
        for r in range(ROWS):
            if grid[r][c]:
                seen_block = True
            elif seen_block:
                holes += 1
    return holes


def bumpiness(heights):
    return sum(abs(heights[i] - heights[i + 1]) for i in range(len(heights) - 1))


def aggregate_height(heights):
    return sum(heights)


def max_height(heights):
    return max(heights) if heights else 0


def right_well_features(grid, heights):
    right = COLS - 1
    depth = 0
    blocked = 0
    open_ok = 1

    for r in range(ROWS - 1, -1, -1):
        if grid[r][right]:
            blocked += 1
            if depth == 0:
                open_ok = 0
        else:
            left_filled = (right - 1 >= 0 and bool(grid[r][right - 1]))
            if left_filled:
                depth += 1
            elif depth > 0:
                break

    return {
        "right_well_open": 1.0 if open_ok else 0.0,
        "right_well_depth": float(depth),
        "right_well_blocked": float(blocked),
    }


def perfect_clear(grid):
    return all(not cell for row in grid for cell in row)


def perfect_clear_setup(grid, heights):
    holes = hole_count(grid, heights)
    agg = aggregate_height(heights)
    flatness = -bumpiness(heights)
    empty = sum(1 for row in grid for cell in row if not cell)
    return max(0.0, (empty / (ROWS * COLS)) * 6.0 + flatness * 0.05 - holes * 1.5 - agg * 0.03)


def is_i_ready(next_queue, hold_kind):
    return (next_queue and next_queue[0] == "I") or hold_kind == "I"


@dataclass
class SearchState:
    grid: list
    current_kind: str
    next_queue: tuple
    hold_kind: str | None
    can_hold: bool


class TetrisEnv:
    def __init__(self, render=False, render_fps=FPS, max_pieces=5000):
        self.render_mode = render
        self.render_fps = render_fps
        self.max_pieces = max_pieces

        self._screen = None
        self._clock = None
        self._fonts = None

        self.search_nodes = 0
        self.cache_hits = 0
        self.ttable = {}

        self.reset()

    def _new_bag(self):
        bag = PIECE_KINDS[:]
        random.shuffle(bag)
        return bag

    def _ensure_queue(self, count=7):
        while len(self.queue) < count:
            self.queue.extend(self._new_bag())

    def _pop_kind(self):
        self._ensure_queue()
        return self.queue.pop(0)

    def reset(self):
        self.board = Board()
        self.stats = Stats()
        self.queue = self._new_bag()
        self.current = Piece(self._pop_kind())
        self.next = Piece(self._pop_kind())
        self.hold_kind = None
        self.can_hold = True
        self.done = False
        self.ttable.clear()
        return self._state()

    def _state(self):
        return SearchState(
            grid=clone_grid(self.board.grid),
            current_kind=self.current.kind,
            next_queue=(self.next.kind, *tuple(self.queue[:5])),
            hold_kind=self.hold_kind,
            can_hold=self.can_hold,
        )

    def _spawn_next_on_real_game(self):
        self.current = self.next
        self.next = Piece(self._pop_kind())
        self.can_hold = True

    def _enumerate_actions(self, state):
        actions = []

        def add_for_kind(kind, use_hold):
            for rot_idx, matrix_t in enumerate(unique_rotations(kind)):
                matrix = [list(row) for row in matrix_t]
                width = len(matrix[0])
                for col in range(COLS - width + 1):
                    row = drop_row(state.grid, matrix, col)
                    if row is None:
                        continue
                    actions.append({
                        "kind": kind,
                        "rotation": rot_idx,
                        "matrix": matrix,
                        "col": col,
                        "row": row,
                        "use_hold": use_hold,
                    })

        add_for_kind(state.current_kind, False)

        if state.can_hold:
            held = state.hold_kind if state.hold_kind is not None else (state.next_queue[0] if state.next_queue else None)
            if held is not None:
                add_for_kind(held, True)

        return actions

    def _advance_after_action(self, state, action, new_grid):
        next_queue = list(state.next_queue)
        hold_kind = state.hold_kind
        can_hold = True

        if action["use_hold"]:
            if state.hold_kind is None:
                hold_kind = state.current_kind
                current_kind = next_queue.pop(0)
            else:
                hold_kind = state.current_kind
                current_kind = state.hold_kind
        else:
            current_kind = next_queue.pop(0)

        return SearchState(
            grid=new_grid,
            current_kind=current_kind,
            next_queue=tuple(next_queue),
            hold_kind=hold_kind,
            can_hold=can_hold,
        )

    def _apply_action_to_state(self, state, action):
        color = True
        placed = place_matrix(state.grid, action["matrix"], action["col"], action["row"], color=color)
        cleared_grid, lines = clear_lines(placed)
        next_state = self._advance_after_action(state, action, cleared_grid)
        return next_state, lines

    def _heuristic_score(self, grid, lines, weights, next_queue, hold_kind, bot):
        heights = board_heights(grid)
        holes = hole_count(grid, heights)
        agg = aggregate_height(heights)
        bump = bumpiness(heights)
        mxh = max_height(heights)
        well = right_well_features(grid, heights)
        pc = 1.0 if perfect_clear(grid) else 0.0
        pc_setup = perfect_clear_setup(grid, heights)
        danger = max(0, mxh - (ROWS - 6))
        i_bonus = 1.0 if is_i_ready(next_queue, hold_kind) else 0.0

        score = 0.0
        score += weights["completed_lines"] * lines
        score += weights["holes"] * holes
        score += weights["aggregate_height"] * agg
        score += weights["bumpiness"] * bump
        score += weights["max_height"] * mxh
        score += weights["right_well_open"] * well["right_well_open"]
        score += weights["right_well_depth"] * well["right_well_depth"]
        score += weights["right_well_blocked"] * well["right_well_blocked"]
        score += weights["i_ready_bonus"] * i_bonus
        score += weights["perfect_clear_bonus"] * pc
        score += weights["perfect_clear_setup"] * pc_setup
        score += weights["danger"] * danger

        if bot in ("tree", "perfect_clear"):
            if well["right_well_open"] <= 0 and well["right_well_depth"] < 3:
                score += weights["well_lock_penalty"]

        if bot == "perfect_clear":
            score += pc_setup * 4.0
            score -= well["right_well_depth"] * 0.2

        return score

    def _state_key(self, state, depth, bot):
        grid_key = tuple(tuple(1 if cell else 0 for cell in row) for row in state.grid)
        return (
            grid_key,
            state.current_kind,
            state.next_queue,
            state.hold_kind,
            state.can_hold,
            depth,
            bot,
        )

    def _search_value(self, state, weights, depth, beam, discount, bot):
        key = self._state_key(state, depth, bot)
        if key in self.ttable:
            self.cache_hits += 1
            return self.ttable[key]

        self.search_nodes += 1
        actions = self._enumerate_actions(state)
        if not actions:
            self.ttable[key] = -1e9
            return -1e9

        scored = []
        for action in actions:
            next_state, lines = self._apply_action_to_state(state, action)
            score = self._heuristic_score(
                next_state.grid,
                lines,
                weights,
                next_state.next_queue,
                next_state.hold_kind,
                bot,
            )
            scored.append((score, action, next_state, lines))

        scored.sort(key=lambda x: x[0], reverse=True)
        scored = scored[:beam]

        if depth <= 1:
            best = scored[0][0]
            self.ttable[key] = best
            return best

        best = -1e9
        for immediate, action, next_state, lines in scored:
            future = self._search_value(next_state, weights, depth - 1, beam, discount, bot)
            total = immediate + discount * future
            if total > best:
                best = total

        self.ttable[key] = best
        return best

    def get_best_action(self, bot, weights, depth, beam, discount):
        self.ttable.clear()
        self.search_nodes = 0
        self.cache_hits = 0

        state = self._state()
        actions = self._enumerate_actions(state)
        if not actions:
            return None

        scored = []
        for action in actions:
            next_state, lines = self._apply_action_to_state(state, action)
            immediate = self._heuristic_score(
                next_state.grid,
                lines,
                weights,
                next_state.next_queue,
                next_state.hold_kind,
                bot,
            )

            if bot == "greedy" or depth <= 1:
                total = immediate
            else:
                future = self._search_value(next_state, weights, depth - 1, beam, discount, bot)
                total = immediate + discount * future

            scored.append((total, action))

        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[0][1]

    def step(self, action, render=False):
        if self.done or action is None:
            self.done = True
            return None, -100.0, True, self._info()

        if action["use_hold"]:
            if self.hold_kind is None:
                self.hold_kind = self.current.kind
                self.current = self.next
                self.next = Piece(self._pop_kind())
            else:
                held = self.hold_kind
                self.hold_kind = self.current.kind
                self.current = Piece(held)

        self.current.matrix = [row[:] for row in action["matrix"]]
        self.current.x = action["col"]
        self.current.y = action["row"]

        self.board.place(self.current)
        cleared = self.board.clear_lines()

        self.stats.add_lines(cleared, self.stats.level)
        self.stats.record()

        if self.stats.pieces >= self.max_pieces:
            self.done = True

        if not self.done:
            self._spawn_next_on_real_game()
            test_state = self._state()
            if not self._enumerate_actions(test_state):
                self.done = True

        reward = float(cleared * 100 + 1)
        if self.done:
            reward -= 50.0

        if self.render_mode or render:
            self._render()

        return None, reward, self.done, self._info()

    def _info(self):
        return {
            "score": self.stats.score,
            "lines": self.stats.lines,
            "level": self.stats.level,
            "pieces": self.stats.pieces,
            "cache_hits": self.cache_hits,
            "nodes": self.search_nodes,
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
        pygame.display.set_caption("Tetris AI")
        self._clock = pygame.time.Clock()
        self._fonts = (
            pygame.font.SysFont("menlo", 28, bold=True),
            pygame.font.SysFont("menlo", 20),
            pygame.font.SysFont("menlo", 14),
        )

    def _render(self, bot_name="tree"):
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
            gy = self.board.ghost_y(self.current)
            draw_piece(board_surf, self.current, ghost_y=gy)

        self._screen.blit(board_surf, (0, 0))
        draw_panel(
            self._screen,
            self.stats,
            self.next,
            self.hold_kind,
            *self._fonts,
            bot_name=bot_name,
        )
        pygame.display.flip()
        self._clock.tick(self.render_fps)
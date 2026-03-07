
import os
import sys
import math
import random
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(__file__))

import pygame

from game import (
    Board, Piece, Stats, TETROMINOES, PIECE_COLORS,
    COLS, ROWS, BOARD_W, BOARD_H, WINDOW_W, WINDOW_H, BG, FPS,
    draw_board, draw_piece, draw_panel,
)

KINDS = tuple(TETROMINOES.keys())
KIND_INDEX = {k: i for i, k in enumerate(KINDS)}

@dataclass(frozen=True)
class State:
    grid: Tuple[Tuple[int, ...], ...]
    current_kind: str
    next_queue: Tuple[str, ...]
    hold_kind: Optional[str]
    hold_used: bool
    score: int
    lines: int
    pieces: int
    well_locked: bool

def empty_grid():
    return tuple(tuple(0 for _ in range(COLS)) for _ in range(ROWS))

def grid_to_mutable(grid):
    return [list(row) for row in grid]

def mutable_to_grid(grid):
    return tuple(tuple(int(v) for v in row) for row in grid)

@lru_cache(maxsize=None)
def rotate_matrix(kind: str, times: int):
    m = tuple(tuple(v for v in row) for row in TETROMINOES[kind])
    for _ in range(times % 4):
        m = tuple(tuple(row) for row in zip(*m[::-1]))
    return m

@lru_cache(maxsize=None)
def rotations_for_kind(kind: str):
    out = []
    seen = set()
    for rot in range(4):
        m = rotate_matrix(kind, rot)
        if m not in seen:
            seen.add(m)
            out.append(m)
    return tuple(out)

def valid_position(grid, matrix, col, row):
    for r, mrow in enumerate(matrix):
        for c, val in enumerate(mrow):
            if not val:
                continue
            nx, ny = col + c, row + r
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

def place_on_grid(grid, matrix, col, row, value=1):
    g = grid_to_mutable(grid)
    for r, mrow in enumerate(matrix):
        for c, val in enumerate(mrow):
            if val:
                ny, nx = row + r, col + c
                if 0 <= ny < ROWS and 0 <= nx < COLS:
                    g[ny][nx] = value
    return mutable_to_grid(g)

def clear_lines(grid):
    g = [list(row) for row in grid if not all(row)]
    cleared = ROWS - len(g)
    while len(g) < ROWS:
        g.insert(0, [0] * COLS)
    return mutable_to_grid(g), cleared

def column_heights(grid):
    h = [0] * COLS
    for c in range(COLS):
        for r in range(ROWS):
            if grid[r][c]:
                h[c] = ROWS - r
                break
    return h

def count_holes(grid, heights=None):
    heights = heights or column_heights(grid)
    holes = 0
    holes_by_col = [0] * COLS
    for c in range(COLS):
        block_seen = False
        for r in range(ROWS):
            if grid[r][c]:
                block_seen = True
            elif block_seen:
                holes += 1
                holes_by_col[c] += 1
    return holes, holes_by_col

def row_transitions(grid):
    total = 0
    for r in range(ROWS):
        prev = 1
        for c in range(COLS):
            cur = 1 if grid[r][c] else 0
            if cur != prev:
                total += 1
            prev = cur
        if prev == 0:
            total += 1
    return total

def col_transitions(grid):
    total = 0
    for c in range(COLS):
        prev = 1
        for r in range(ROWS):
            cur = 1 if grid[r][c] else 0
            if cur != prev:
                total += 1
            prev = cur
        if prev == 0:
            total += 1
    return total

def covered_cells(grid):
    covered = 0
    for c in range(COLS):
        filled_seen = False
        for r in range(ROWS):
            if grid[r][c]:
                filled_seen = True
            elif filled_seen:
                covered += 1
    return covered

def right_well_depth(grid):
    # contiguous empties from bottom in rightmost column with support on left
    depth = 0
    c = COLS - 1
    for r in range(ROWS - 1, -1, -1):
        if grid[r][c]:
            break
        left_filled = (c - 1 >= 0 and grid[r][c - 1])
        if left_filled:
            depth += 1
        else:
            break
    return depth

def right_well_open(grid):
    c = COLS - 1
    # open if top 4 cells of right column are empty
    return int(all(not grid[r][c] for r in range(min(4, ROWS))))

def right_well_blocked(grid):
    c = COLS - 1
    return int(any(grid[r][c] for r in range(0, min(6, ROWS))))

def perfect_clear_possible(grid):
    filled = sum(1 for row in grid for v in row if v)
    heights = column_heights(grid)
    holes, _ = count_holes(grid, heights)
    return int(filled <= 10 and holes == 0 and max(heights, default=0) <= 2)

def perfect_clear(grid):
    return int(all(not v for row in grid for v in row))

def i_ready_bonus(state: State):
    return int(state.current_kind == "I" or (state.next_queue and state.next_queue[0] == "I"))

def compute_features(state: State):
    heights = column_heights(state.grid)
    agg_height = sum(heights)
    holes, holes_by_col = count_holes(state.grid, heights)
    bumpiness = sum(abs(heights[i] - heights[i + 1]) for i in range(COLS - 1))
    max_height = max(heights, default=0)
    rows_t = row_transitions(state.grid)
    cols_t = col_transitions(state.grid)
    covered = covered_cells(state.grid)
    rw_depth = right_well_depth(state.grid)
    rw_open = right_well_open(state.grid)
    rw_blocked = right_well_blocked(state.grid)
    pc_setup = perfect_clear_possible(state.grid)
    pc = perfect_clear(state.grid)
    return {
        "aggregate_height": agg_height,
        "holes": holes,
        "bumpiness": bumpiness,
        "max_height": max_height,
        "row_transitions": rows_t,
        "col_transitions": cols_t,
        "covered_cells": covered,
        "right_well_depth": rw_depth,
        "right_well_open": rw_open,
        "right_well_blocked": rw_blocked,
        "perfect_clear_setup": pc_setup,
        "perfect_clear": pc,
        "i_ready_bonus": i_ready_bonus(state),
    }

def maybe_lock_well(prev_state: State, col: int, matrix) -> Tuple[bool, bool]:
    # keep a right-side well. bonus if we preserve it, penalty if we fill it without strong reason.
    rightmost_used = any(val and (col + c == COLS - 1) for r, row in enumerate(matrix) for c, val in enumerate(row))
    if prev_state.well_locked:
        return True, rightmost_used
    return False, rightmost_used

class TetrisEnv:
    def __init__(self, render=False, render_fps=FPS, seed=None):
        self.render_mode = render
        self.render_fps = render_fps
        self._screen = None
        self._clock = None
        self._fonts = None
        self._rng = random.Random(seed)
        self.state: Optional[State] = None
        self.stats = None
        self.current_piece = None
        self.next_piece = None
        self.hold_piece = None
        self.done = False
        self.node_count = 0
        self.cache_hits = 0
        self.transposition: Dict[Tuple, float] = {}
        self.last_search_summary = {}

    def _new_bag(self):
        bag = list(KINDS)
        self._rng.shuffle(bag)
        return bag

    def _make_initial_state(self):
        bag = self._new_bag()
        current = bag.pop()
        nxt = tuple(bag)
        return State(
            grid=empty_grid(),
            current_kind=current,
            next_queue=nxt,
            hold_kind=None,
            hold_used=False,
            score=0,
            lines=0,
            pieces=0,
            well_locked=True,
        )

    def reset(self):
        self.state = self._make_initial_state()
        self.stats = Stats()
        self.done = False
        self.transposition = {}
        self.node_count = 0
        self.cache_hits = 0
        self._sync_visuals()
        if self.render_mode:
            self._init_render()
        return self._observe(), self._info()

    def _observe(self):
        feats = compute_features(self.state)
        return feats

    def _draw_queue(self):
        if len(self.state.next_queue) >= 2:
            return self.state.next_queue[0]
        return self.state.next_queue[0] if self.state.next_queue else None

    def _ensure_queue(self, q):
        q = list(q)
        while len(q) < 6:
            q.extend(self._new_bag())
        return tuple(q)

    def _advance_queue(self, q):
        q = self._ensure_queue(q)
        current = q[0]
        q = q[1:]
        q = self._ensure_queue(q)
        return current, q

    def _spawn_visual_piece(self, kind):
        p = Piece(kind)
        p.y = 0
        p.x = COLS // 2 - len(p.matrix[0]) // 2
        return p

    def _sync_visuals(self):
        if self.state is None:
            return
        self.current_piece = self._spawn_visual_piece(self.state.current_kind)
        nxt = self.state.next_queue[0] if self.state.next_queue else None
        self.next_piece = self._spawn_visual_piece(nxt) if nxt else None
        self.hold_piece = self._spawn_visual_piece(self.state.hold_kind) if self.state.hold_kind else None

    def _state_to_board(self, state):
        b = Board()
        b.grid = [[PIECE_COLORS["I"] if cell else None for cell in row] for row in state.grid]
        return b

    def _simulate_place(self, state: State, kind: str, matrix, col: int, use_hold=False):
        row = drop_row(state.grid, matrix, col)
        if row is None:
            return None

        placed = place_on_grid(state.grid, matrix, col, row, 1)
        cleared_grid, cleared = clear_lines(placed)

        next_queue = list(state.next_queue)
        hold_kind = state.hold_kind
        hold_used = False

        if use_hold:
            # state.current_kind is not placed; kind is either old hold or queue[0]
            pass

        next_current = next_queue[0] if next_queue else self._new_bag()[0]
        next_queue = tuple(next_queue[1:]) if next_queue else tuple(self._new_bag())

        lock_active, broke_well = maybe_lock_well(state, col, matrix)
        next_state = State(
            grid=cleared_grid,
            current_kind=next_current,
            next_queue=self._ensure_queue(next_queue),
            hold_kind=hold_kind,
            hold_used=False,
            score=state.score + (100 if cleared == 1 else 300 if cleared == 2 else 500 if cleared == 3 else 800 if cleared >= 4 else 0),
            lines=state.lines + cleared,
            pieces=state.pieces + 1,
            well_locked=lock_active and not broke_well,
        )
        return next_state, cleared, row, broke_well

    def _enumerate_actions(self, state: State, use_hold=True):
        actions = []
        kind = state.current_kind

        for rot_idx, matrix in enumerate(rotations_for_kind(kind)):
            width = len(matrix[0])
            for col in range(COLS - width + 1):
                sim = self._simulate_place(state, kind, matrix, col, use_hold=False)
                if sim is not None:
                    next_state, cleared, row, broke_well = sim
                    actions.append({
                        "kind": kind,
                        "rotation": rot_idx,
                        "matrix": matrix,
                        "col": col,
                        "row": row,
                        "use_hold": False,
                        "cleared": cleared,
                        "broke_well": broke_well,
                        "next_state": next_state,
                    })

        if use_hold and not state.hold_used:
            if state.hold_kind is None:
                q = list(state.next_queue)
                if q:
                    swapped_kind = q[0]
                    swapped_q = tuple(self._ensure_queue(tuple(q[1:])))
                    base_hold = state.current_kind
                    hold_state = State(
                        grid=state.grid,
                        current_kind=swapped_kind,
                        next_queue=swapped_q,
                        hold_kind=base_hold,
                        hold_used=True,
                        score=state.score,
                        lines=state.lines,
                        pieces=state.pieces,
                        well_locked=state.well_locked,
                    )
                    for rot_idx, matrix in enumerate(rotations_for_kind(swapped_kind)):
                        width = len(matrix[0])
                        for col in range(COLS - width + 1):
                            sim = self._simulate_place(hold_state, swapped_kind, matrix, col, use_hold=True)
                            if sim is not None:
                                next_state, cleared, row, broke_well = sim
                                next_state = State(
                                    grid=next_state.grid,
                                    current_kind=next_state.current_kind,
                                    next_queue=next_state.next_queue,
                                    hold_kind=base_hold,
                                    hold_used=False,
                                    score=next_state.score,
                                    lines=next_state.lines,
                                    pieces=next_state.pieces,
                                    well_locked=next_state.well_locked,
                                )
                                actions.append({
                                    "kind": swapped_kind,
                                    "rotation": rot_idx,
                                    "matrix": matrix,
                                    "col": col,
                                    "row": row,
                                    "use_hold": True,
                                    "cleared": cleared,
                                    "broke_well": broke_well,
                                    "next_state": next_state,
                                })
            else:
                swapped_kind = state.hold_kind
                base_hold = state.current_kind
                hold_state = State(
                    grid=state.grid,
                    current_kind=swapped_kind,
                    next_queue=state.next_queue,
                    hold_kind=base_hold,
                    hold_used=True,
                    score=state.score,
                    lines=state.lines,
                    pieces=state.pieces,
                    well_locked=state.well_locked,
                )
                for rot_idx, matrix in enumerate(rotations_for_kind(swapped_kind)):
                    width = len(matrix[0])
                    for col in range(COLS - width + 1):
                        sim = self._simulate_place(hold_state, swapped_kind, matrix, col, use_hold=True)
                        if sim is not None:
                            next_state, cleared, row, broke_well = sim
                            next_state = State(
                                grid=next_state.grid,
                                current_kind=next_state.current_kind,
                                next_queue=next_state.next_queue,
                                hold_kind=base_hold,
                                hold_used=False,
                                score=next_state.score,
                                lines=next_state.lines,
                                pieces=next_state.pieces,
                                well_locked=next_state.well_locked,
                            )
                            actions.append({
                                "kind": swapped_kind,
                                "rotation": rot_idx,
                                "matrix": matrix,
                                "col": col,
                                "row": row,
                                "use_hold": True,
                                "cleared": cleared,
                                "broke_well": broke_well,
                                "next_state": next_state,
                            })

        return actions

    def _evaluate_state(self, state: State, action, weights):
        feats = compute_features(state)
        value = 0.0
        for key, wt in weights.items():
            if key in feats:
                value += wt * feats[key]

        cleared = action.get("cleared", 0)
        value += weights.get("completed_lines", 0.0) * cleared
        if cleared == 4:
            value += weights.get("tetris_bonus", 0.0)
        if feats["perfect_clear"]:
            value += weights.get("perfect_clear_bonus", 0.0)
        if state.well_locked and not action.get("broke_well", False):
            value += weights.get("well_lock_bonus", 0.0)
        if action.get("broke_well", False):
            value += weights.get("well_lock_break_penalty", 0.0)
        if feats["max_height"] >= ROWS - 4:
            value += weights.get("danger_penalty", 0.0) * (feats["max_height"] - (ROWS - 5))
        return value

    def _search(self, state: State, depth: int, take: int, discount: float, weights, use_hold=True):
        key = (state.grid, state.current_kind, state.next_queue[:4], state.hold_kind, state.well_locked, depth, take, use_hold)
        if key in self.transposition:
            self.cache_hits += 1
            return self.transposition[key]

        self.node_count += 1
        actions = self._enumerate_actions(state, use_hold=use_hold)
        if not actions:
            return -1e9

        scored = [(self._evaluate_state(a["next_state"], a, weights), a) for a in actions]
        scored.sort(key=lambda t: t[0], reverse=True)
        best_now = scored[0][0]

        if depth <= 1:
            self.transposition[key] = best_now
            return best_now

        best = -1e18
        for base_score, action in scored[:max(1, take)]:
            fut = self._search(action["next_state"], depth - 1, take, discount, weights, use_hold=use_hold)
            total = base_score + discount * fut
            if total > best:
                best = total
        self.transposition[key] = best
        return best

    def get_best_action(self, weights, depth=2, take=8, discount=0.985, use_hold=True):
        self.node_count = 0
        self.cache_hits = 0
        self.transposition = {}

        actions = self._enumerate_actions(self.state, use_hold=use_hold)
        if not actions:
            return None

        scored = []
        for a in actions:
            base = self._evaluate_state(a["next_state"], a, weights)
            total = base
            if depth > 1:
                total += discount * self._search(a["next_state"], depth - 1, take, discount, weights, use_hold=use_hold)
            scored.append((total, base, a))

        scored.sort(key=lambda t: t[0], reverse=True)
        best = scored[0][2]
        self.last_search_summary = {
            "depth": depth,
            "beam": take,
            "nodes": self.node_count,
            "cache_hits": self.cache_hits,
            "well_locked": self.state.well_locked,
        }
        return (best["rotation"], best["col"], best["use_hold"])

    def step(self, action, render=False):
        if action is None:
            self.done = True
            return self._observe(), -999.0, True, self._info()

        rotation, col, use_hold = action
        chosen = None
        for a in self._enumerate_actions(self.state, use_hold=True):
            if a["rotation"] == rotation and a["col"] == col and a["use_hold"] == use_hold:
                chosen = a
                break
        if chosen is None:
            self.done = True
            return self._observe(), -999.0, True, self._info()

        self.state = chosen["next_state"]
        self.stats.score = self.state.score
        self.stats.lines = self.state.lines
        self.stats.pieces = self.state.pieces
        self.stats.level = self.stats.lines // 10 + 1
        self.stats.record()
        self._sync_visuals()

        reward = self._evaluate_state(self.state, chosen, {
            "aggregate_height": -0.15,
            "holes": -1.25,
            "bumpiness": -0.08,
            "max_height": -0.05,
            "row_transitions": -0.03,
            "col_transitions": -0.03,
            "covered_cells": -0.02,
            "right_well_depth": 0.3,
            "right_well_open": 0.4,
            "right_well_blocked": -0.8,
            "perfect_clear_setup": 0.3,
            "perfect_clear_bonus": 8.0,
            "well_lock_bonus": 0.6,
            "well_lock_break_penalty": -1.0,
            "completed_lines": 1.1,
            "tetris_bonus": 1.2,
            "danger_penalty": -0.6,
            "i_ready_bonus": 0.2,
        })

        if not self._enumerate_actions(self.state, use_hold=True):
            self.done = True
            reward -= 25.0

        if self.render_mode or render:
            self._render()
        return self._observe(), reward, self.done, self._info()

    def _info(self):
        return {
            "score": self.stats.score if self.stats else 0,
            "lines": self.stats.lines if self.stats else 0,
            "level": self.stats.level if self.stats else 1,
            "pieces": self.stats.pieces if self.stats else 0,
            **self.last_search_summary,
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
        pygame.display.set_caption("Tetris Search Bot")
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
                raise SystemExit

        self._screen.fill(BG)
        board_surf = pygame.Surface((BOARD_W, BOARD_H))
        board_surf.fill(BG)
        board = self._state_to_board(self.state)
        draw_board(board_surf, board)
        if self.current_piece:
            gy = board.ghost_y(self.current_piece)
            draw_piece(board_surf, self.current_piece, ghost_y=gy)
        self._screen.blit(board_surf, (0, 0))
        draw_panel(
            self._screen,
            self.stats,
            self.state.next_queue[0] if self.state.next_queue else None,
            self.state.hold_kind,
            self._fonts,
            extra=self.last_search_summary,
        )
        pygame.display.flip()
        self._clock.tick(self.render_fps)

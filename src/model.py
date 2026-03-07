
import json
import os

DEFAULT_WEIGHTS = {
    "aggregate_height": -0.52,
    "holes": -6.8,
    "bumpiness": -0.32,
    "completed_lines": 7.4,
    "max_height": -0.30,
    "row_transitions": -0.28,
    "col_transitions": -0.25,
    "covered_cells": -0.22,
    "right_well_depth": 1.75,
    "right_well_open": 2.2,
    "right_well_blocked": -4.4,
    "well_lock_bonus": 2.8,
    "well_lock_break_penalty": -5.8,
    "i_ready_bonus": 2.6,
    "perfect_clear_bonus": 18.0,
    "perfect_clear_setup": 1.35,
    "tetris_bonus": 4.6,
    "danger_penalty": -1.1,
}

WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "heuristic_weights.json")

def load_weights():
    if os.path.exists(WEIGHTS_PATH):
        try:
            with open(WEIGHTS_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            merged = dict(DEFAULT_WEIGHTS)
            merged.update({k: float(v) for k, v in data.items()})
            return merged
        except Exception:
            pass
    return dict(DEFAULT_WEIGHTS)

def save_weights(weights):
    os.makedirs(os.path.dirname(WEIGHTS_PATH), exist_ok=True)
    with open(WEIGHTS_PATH, "w", encoding="utf-8") as f:
        json.dump(weights, f, indent=2, sort_keys=True)

DEFAULT_WEIGHTS = {
    "holes": -7.5,
    "aggregate_height": -0.55,
    "bumpiness": -0.30,
    "completed_lines": 6.5,
    "max_height": -0.20,
    "right_well_open": 2.5,
    "right_well_depth": 1.1,
    "right_well_blocked": -4.0,
    "i_ready_bonus": 2.2,
    "perfect_clear_bonus": 35.0,
    "perfect_clear_setup": 4.0,
    "danger": -3.0,
    "well_lock_penalty": -6.5,
}

BOT_PRESETS = {
    "greedy": {
        **DEFAULT_WEIGHTS,
        "completed_lines": 5.5,
        "right_well_open": 1.6,
        "perfect_clear_setup": 1.0,
    },
    "tree": {
        **DEFAULT_WEIGHTS,
    },
    "perfect_clear": {
        **DEFAULT_WEIGHTS,
        "completed_lines": 4.2,
        "perfect_clear_bonus": 60.0,
        "perfect_clear_setup": 8.0,
        "holes": -9.0,
        "aggregate_height": -0.45,
    },
}
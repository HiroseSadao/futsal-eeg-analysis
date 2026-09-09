"""Read fixed trial orders and permutation seeds from external study metadata."""

from __future__ import annotations

import json
from pathlib import Path

from common import CHANNELS, DAYS, DAY_TRIALS


def validate_design(document: dict, n_permutations: int) -> dict:
    """Validate trial orders and permutation seeds."""
    if type(n_permutations) is not int or n_permutations < 1:
        raise ValueError("At least one permutation is required")
    if not isinstance(document, dict) or document.get("version") != 1:
        raise ValueError("Expected SVM design version 1")
    entries = document.get("tasks")
    if not isinstance(entries, list) or not entries:
        raise ValueError("The SVM design must contain a non-empty task list")

    design = {}
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("Each design task must be an object")
        participant_id = entry.get("participant_id")
        day = entry.get("day")
        channel = entry.get("channel")
        if not isinstance(participant_id, str) or not participant_id.strip():
            raise ValueError("Each design task requires a participant_id")
        if day not in DAYS or channel not in CHANNELS:
            raise ValueError("Invalid day or channel in SVM design")
        key = (participant_id, day, channel)
        if key in design:
            raise ValueError(f"Duplicate SVM design task: {key}")

        order = entry.get("trial_order")
        if (
            not isinstance(order, list)
            or any(type(number) is not int for number in order)
            or sorted(order) != list(DAY_TRIALS[day])
        ):
            raise ValueError(f"Expected each recorded trial exactly once: {key}")

        for field in ("within_seeds", "cross_seeds"):
            seeds = entry.get(field)
            if (
                not isinstance(seeds, list)
                or len(seeds) < n_permutations
                or any(
                    type(seed) is not int or not 0 <= seed < 2**32
                    for seed in seeds
                )
            ):
                raise ValueError(f"Invalid or insufficient {field} for {key}")
        design[key] = entry
    return design


def load_design(path: Path, n_permutations: int) -> dict:
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} is missing. Supply the recorded SVM design; "
            "trial order and randomization are not inferred during analysis."
        )
    with path.open(encoding="utf-8") as handle:
        return validate_design(json.load(handle), n_permutations)

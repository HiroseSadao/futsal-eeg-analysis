"""Record trial order and permutation seeds for the SVM analyses.

Capture the source filesystem's trial order before reorganizing its files.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd

from common import (
    CHANNELS,
    DAYS,
    PROJECT_ROOT,
    load_participants,
    parse_trial_number,
    stable_seed,
)
from svm_design import validate_design


def capture_design(
    processed_root: Path, participants_path: Path, n_permutations: int
) -> dict:
    participants = load_participants(participants_path)
    metadata = pd.read_csv(participants_path, dtype=str)
    if (
        "randomization_key" not in metadata
        or metadata["randomization_key"].isna().any()
    ):
        raise ValueError(
            "The participant table must include each original randomization_key"
        )
    keys = dict(zip(metadata["participant_id"], metadata["randomization_key"]))
    if any(not value.strip() for value in keys.values()):
        raise ValueError("randomization_key values must not be empty")
    if len(set(keys.values())) != len(keys):
        raise ValueError("randomization_key values must be unique")

    entries = []
    for participant in participants.itertuples(index=False):
        key = keys[participant.participant_id]
        for day in DAYS:
            folder = processed_root / participant.directory / f"{day}_all"
            # Preserve the filesystem's trial order.
            filenames = os.listdir(folder)
            test_day = "day2" if day == "day1" else "day1"
            for channel in CHANNELS:
                order = [
                    parse_trial_number(filename)
                    for filename in filenames
                    if filename.endswith(".csv")
                    and f"_filtered_{channel}_" in filename
                ]
                entries.append(
                    {
                        "participant_id": participant.participant_id,
                        "day": day,
                        "channel": channel,
                        "trial_order": order,
                        # Store numeric seeds without participant keys.
                        "within_seeds": [
                            stable_seed("svm-label-shuffle", channel, key, day, index)
                            for index in range(n_permutations)
                        ],
                        "cross_seeds": [
                            stable_seed(
                                "svm-crossday-target-label",
                                channel, key, day, test_day, index,
                            )
                            for index in range(n_permutations)
                        ],
                    }
                )
    document = {"version": 1, "tasks": entries}
    validate_design(document, n_permutations)
    return document


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-root", type=Path, required=True)
    parser.add_argument(
        "--participants", type=Path, default=PROJECT_ROOT / "data/participants.csv"
    )
    parser.add_argument(
        "--output", type=Path, default=PROJECT_ROOT / "data/svm_design.json"
    )
    parser.add_argument("--permutations", type=int, default=199)
    args = parser.parse_args()
    document = capture_design(args.processed_root, args.participants, args.permutations)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Keep existing designs unchanged.
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(document, handle, indent=2)
        handle.write("\n")


if __name__ == "__main__":
    main()

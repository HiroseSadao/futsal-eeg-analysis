"""Summarize free-kick outcomes by participant and recording day."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from common import PROJECT_ROOT, load_outcomes, load_participants


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--participants",
        type=Path,
        default=PROJECT_ROOT / "data/participants.csv",
    )
    parser.add_argument(
        "--outcomes",
        type=Path,
        default=PROJECT_ROOT / "data/trial_outcomes.csv",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=PROJECT_ROOT / "results/behavior",
    )
    args = parser.parse_args()

    participants = load_participants(args.participants)
    outcomes = load_outcomes(args.outcomes)
    unexpected = sorted(
        set(outcomes["participant_id"]) - set(participants["participant_id"])
    )
    if unexpected:
        raise ValueError(f"Unknown participants in outcome table: {unexpected}")

    rates = (
        outcomes.groupby(["participant_id", "day"], sort=True)["outcome"]
        .agg(n_trials="size", n_success="sum")
        .reset_index()
    )
    rates["n_failure"] = rates["n_trials"] - rates["n_success"]
    rates["success_rate"] = rates["n_success"] / rates["n_trials"]
    if len(rates) != len(participants) * 2 or not (rates["n_trials"] == 50).all():
        raise ValueError("Expected 50 outcomes per participant and day")

    rows = []
    groups = [(day, group) for day, group in rates.groupby("day")]
    groups.append(("all_participant_days", rates))
    for label, group in groups:
        values = group["success_rate"].to_numpy(float)
        rows.append(
            {
                "group": label,
                "n_participant_days": values.size,
                "mean_success_rate": np.mean(values),
                "sample_sd": np.std(values, ddof=1),
                "median_success_rate": np.median(values),
                "q1_success_rate": np.quantile(values, 0.25),
                "q3_success_rate": np.quantile(values, 0.75),
                "minimum_success_rate": np.min(values),
                "maximum_success_rate": np.max(values),
            }
        )

    args.results_dir.mkdir(parents=True, exist_ok=True)
    rates.to_csv(args.results_dir / "participant_day_rates.csv", index=False)
    pd.DataFrame(rows).to_csv(
        args.results_dir / "success_rate_summary.csv", index=False
    )


if __name__ == "__main__":
    main()

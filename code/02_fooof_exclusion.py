"""Fit FOOOF to every trial and write the trial-exclusion table."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from fooof import FOOOF

from common import (
    CHANNELS,
    N_SAMPLES,
    PROJECT_ROOT,
    SAMPLE_RATE_HZ,
    find_processed_trial,
    load_participants,
    read_waveform,
)


def fit_fooof(waveform: np.ndarray) -> dict[str, float]:
    coefficients = np.fft.rfft(waveform)
    frequencies = np.fft.rfftfreq(waveform.size, d=1.0 / SAMPLE_RATE_HZ)
    power = (np.abs(coefficients) / (N_SAMPLES / 2.0)) ** 2

    model = FOOOF(
        peak_width_limits=(2.0, 8.0),
        max_n_peaks=6,
        min_peak_height=0.15,
        verbose=False,
    )
    model.fit(frequencies, power, (3.0, 70.0))
    return {
        "aperiodic_offset": float(model.aperiodic_params_[0]),
        "aperiodic_exponent": float(model.aperiodic_params_[1]),
        "fit_r_squared": float(model.r_squared_),
        "fit_error": float(model.error_),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--processed-root", type=Path, default=PROJECT_ROOT / "data/processed"
    )
    parser.add_argument(
        "--participants",
        type=Path,
        default=PROJECT_ROOT / "data/participants.csv",
    )
    parser.add_argument(
        "--metrics",
        type=Path,
        default=PROJECT_ROOT / "results/fooof_metrics.csv",
    )
    parser.add_argument(
        "--exclusions",
        type=Path,
        default=PROJECT_ROOT / "results/excluded_trials.csv",
    )
    args = parser.parse_args()

    rows = []
    for participant in load_participants(args.participants).itertuples(index=False):
        for number in range(1, 101):
            day = "day1" if number <= 50 else "day2"
            for channel in CHANNELS:
                path = find_processed_trial(
                    args.processed_root,
                    participant.directory,
                    day,
                    channel,
                    number,
                )
                rows.append(
                    {
                        "participant_id": participant.participant_id,
                        "trial_number": number,
                        "day": day,
                        "channel": channel,
                        **fit_fooof(read_waveform(path, dtype="float64")),
                    }
                )

    metrics = pd.DataFrame(rows).sort_values(
        ["participant_id", "trial_number", "channel"]
    )
    negative = metrics.loc[metrics["aperiodic_exponent"] < 0].copy()
    exclusions = negative.assign(
        reason="negative_aperiodic_exponent",
        reference_channel=negative["channel"],
    ).loc[
        :,
        ["participant_id", "trial_number", "reason", "reference_channel"],
    ]
    exclusions = exclusions.drop_duplicates(
        ["participant_id", "trial_number"]
    ).sort_values(["participant_id", "trial_number"])

    args.metrics.parent.mkdir(parents=True, exist_ok=True)
    args.exclusions.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(args.metrics, index=False)
    exclusions.to_csv(args.exclusions, index=False)


if __name__ == "__main__":
    main()

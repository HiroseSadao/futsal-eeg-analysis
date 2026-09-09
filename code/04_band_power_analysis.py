"""Compare alpha and beta squared FFT magnitudes between kick outcomes."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.fft import rfft, rfftfreq
from scipy.stats import shapiro, ttest_rel, wilcoxon

from common import (
    CHANNELS,
    PROJECT_ROOT,
    SAMPLE_RATE_HZ,
    find_processed_trial,
    holm_adjust,
    load_exclusions,
    load_outcomes,
    load_participants,
    read_waveform,
)


BANDS_HZ = {"alpha": (8.0, 15.0), "beta": (15.0, 30.0)}


def band_value(waveform: np.ndarray, limits: tuple[float, float]) -> float:
    """Mean unnormalized squared FFT magnitude within a frequency band."""
    frequencies = rfftfreq(waveform.size, d=1.0 / SAMPLE_RATE_HZ)
    squared_magnitude = np.abs(rfft(waveform)) ** 2
    mask = (frequencies >= limits[0]) & (frequencies <= limits[1])
    return float(np.mean(squared_magnitude[mask]))


def condition_average(
    processed_root: Path,
    directory: str,
    participant_id: str,
    channel: str,
    trial_numbers: list[int],
    exclusions: set[tuple[str, int]],
) -> tuple[np.ndarray, int]:
    waveforms = []
    for number in trial_numbers:
        if (participant_id, number) in exclusions:
            continue
        day = "day1" if number <= 50 else "day2"
        path = find_processed_trial(
            processed_root, directory, day, channel, number
        )
        waveforms.append(read_waveform(path, dtype="float64"))
    if not waveforms:
        raise ValueError(f"No waveforms for {participant_id}/{channel}")
    return np.mean(np.vstack(waveforms), axis=0), len(waveforms)


def collect_values(args: argparse.Namespace) -> pd.DataFrame:
    participants = load_participants(args.participants)
    outcomes = load_outcomes(args.outcomes)
    exclusions = load_exclusions(args.exclusions)
    rows = []

    for participant in participants.itertuples(index=False):
        participant_outcomes = outcomes.loc[
            outcomes["participant_id"] == participant.participant_id
        ]
        success_trials = participant_outcomes.loc[
            participant_outcomes["outcome"] == 1, "trial_number"
        ].astype(int).tolist()
        failure_trials = participant_outcomes.loc[
            participant_outcomes["outcome"] == 0, "trial_number"
        ].astype(int).tolist()

        for channel in CHANNELS:
            success, n_success = condition_average(
                args.processed_root,
                participant.directory,
                participant.participant_id,
                channel,
                success_trials,
                exclusions,
            )
            failure, n_failure = condition_average(
                args.processed_root,
                participant.directory,
                participant.participant_id,
                channel,
                failure_trials,
                exclusions,
            )
            for band, limits in BANDS_HZ.items():
                success_value = band_value(success, limits)
                failure_value = band_value(failure, limits)
                rows.append(
                    {
                        "participant_id": participant.participant_id,
                        "channel": channel,
                        "band": band,
                        "n_success_trials": n_success,
                        "n_failure_trials": n_failure,
                        "success_value": success_value,
                        "failure_value": failure_value,
                        "difference": success_value - failure_value,
                    }
                )
    return pd.DataFrame(rows).sort_values(["band", "channel", "participant_id"])


def summarize(values: pd.DataFrame) -> pd.DataFrame:
    """Use two-sided paired tests and Holm correction within each band."""
    rows = []
    for (band, channel), group in values.groupby(["band", "channel"]):
        success = group["success_value"].to_numpy(float)
        failure = group["failure_value"].to_numpy(float)
        differences = success - failure
        normality = shapiro(differences)
        if normality.pvalue > 0.05:
            test = ttest_rel(success, failure, alternative="two-sided")
            test_name = "paired_t_test_two_sided"
        else:
            test = wilcoxon(success, failure, alternative="two-sided")
            test_name = "wilcoxon_signed_rank_two_sided"
        rows.append(
            {
                "band": band,
                "channel": channel,
                "n_participants": group["participant_id"].nunique(),
                "mean_success": np.mean(success),
                "mean_failure": np.mean(failure),
                "mean_difference": np.mean(differences),
                "shapiro_w": normality.statistic,
                "shapiro_p": normality.pvalue,
                "test_name": test_name,
                "test_statistic": test.statistic,
                "p_value_two_sided": test.pvalue,
                "p_value_holm_within_band": np.nan,
            }
        )

    summary = pd.DataFrame(rows)
    for band in BANDS_HZ:
        mask = summary["band"] == band
        summary.loc[mask, "p_value_holm_within_band"] = holm_adjust(
            summary.loc[mask, "p_value_two_sided"].to_numpy(float)
        )
    summary["significant_holm_0_05"] = (
        summary["p_value_holm_within_band"] < 0.05
    )
    return summary.sort_values(["band", "channel"])


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
        "--outcomes",
        type=Path,
        default=PROJECT_ROOT / "data/trial_outcomes.csv",
    )
    parser.add_argument(
        "--exclusions",
        type=Path,
        default=PROJECT_ROOT / "results/excluded_trials.csv",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=PROJECT_ROOT / "results/band_power",
    )
    args = parser.parse_args()

    values = collect_values(args)
    summary = summarize(values)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    values.to_csv(args.results_dir / "participant_values.csv", index=False)
    summary.to_csv(args.results_dir / "statistics.csv", index=False)


if __name__ == "__main__":
    main()

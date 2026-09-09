"""Compute Morlet-wavelet power of the mean pre-movement EEG waveform."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pywt

from common import (
    CHANNELS,
    N_SAMPLES,
    PROJECT_ROOT,
    SAMPLE_RATE_HZ,
    find_processed_trial,
    load_exclusions,
    load_participants,
    read_waveform,
)


WAVELET = "cmor1.5-1.0"
FREQUENCIES_HZ = np.linspace(1.0, 50.0, 50)


def mean_wavelet_power(
    waveform: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Transform the trial-averaged waveform, then square its magnitude."""
    if waveform.shape != (N_SAMPLES,) or not np.isfinite(waveform).all():
        raise ValueError(f"Expected a finite waveform with {N_SAMPLES} samples")
    # Frequencies are equally spaced; the corresponding scales are not.
    # cmor1.5-1.0 has a center frequency of 1.
    scales = SAMPLE_RATE_HZ / FREQUENCIES_HZ
    coefficients, frequencies = pywt.cwt(
        waveform, scales, WAVELET, sampling_period=1.0 / SAMPLE_RATE_HZ
    )
    return frequencies, scales, np.abs(coefficients) ** 2


def analyze(args: argparse.Namespace) -> dict[str, dict]:
    participants = load_participants(args.participants)
    if participants.empty:
        raise ValueError("The participant table is empty")
    exclusions = load_exclusions(args.exclusions)
    results = {}
    for channel in CHANNELS:
        wave_sum = np.zeros(N_SAMPLES)
        trial_count = 0
        for participant in participants.itertuples(index=False):
            participant_trials = 0
            for number in range(1, 101):
                if (participant.participant_id, number) in exclusions:
                    continue
                day = "day1" if number <= 50 else "day2"
                path = find_processed_trial(
                    args.processed_root, participant.directory, day, channel, number
                )
                # Load the preprocessed 2-second pre-movement segment.
                wave_sum += read_waveform(path, dtype="float64")
                participant_trials += 1
            if participant_trials == 0:
                raise ValueError(f"No retained trials for {participant.participant_id}")
            trial_count += participant_trials

        # Average retained trials with equal weights before the transform.
        mean_waveform = wave_sum / trial_count
        frequencies, scales, power = mean_wavelet_power(mean_waveform)
        results[channel] = {
            "time_relative_to_onset_s": (
                np.arange(N_SAMPLES) - N_SAMPLES
            ) / SAMPLE_RATE_HZ,
            "frequency_hz": frequencies,
            "scales": scales,
            "mean_waveform_v": mean_waveform,
            "power_v2": power,
            "n_trials": trial_count,
            "n_participants": len(participants),
        }
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--processed-root", type=Path, default=PROJECT_ROOT / "data/processed"
    )
    parser.add_argument(
        "--participants", type=Path, default=PROJECT_ROOT / "data/participants.csv"
    )
    parser.add_argument(
        "--exclusions", type=Path, default=PROJECT_ROOT / "results/excluded_trials.csv"
    )
    parser.add_argument(
        "--results-dir", type=Path, default=PROJECT_ROOT / "results/cwt"
    )
    args = parser.parse_args()
    results = analyze(args)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    for channel, result in results.items():
        np.savez_compressed(args.results_dir / f"{channel}_mean_waveform_cwt.npz", **result)
        pd.DataFrame(
            {
                "time_relative_to_onset_s": result["time_relative_to_onset_s"],
                "mean_waveform_v": result["mean_waveform_v"],
            }
        ).to_csv(args.results_dir / f"{channel}_mean_waveform.csv", index=False)


if __name__ == "__main__":
    main()

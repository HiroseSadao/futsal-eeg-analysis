"""Fit FOOOF to participant-mean spectra and summarize spectral fit quality."""

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
    load_exclusions,
    load_participants,
    read_waveform,
)


FIT_RANGE_HZ = (3.0, 70.0)


def trial_power_spectrum(waveform: np.ndarray) -> np.ndarray:
    """Compute the normalized single-trial FFT power spectrum."""
    return (np.abs(np.fft.rfft(waveform)) / (N_SAMPLES / 2.0)) ** 2


def fit_spectrum(frequencies: np.ndarray, power: np.ndarray) -> FOOOF:
    model = FOOOF(
        peak_width_limits=(2.0, 8.0),
        max_n_peaks=6,
        min_peak_height=0.15,
        verbose=False,
    )
    model.fit(frequencies, power, FIT_RANGE_HZ)
    if not np.isfinite([model.r_squared_, model.error_]).all():
        raise ValueError("FOOOF did not produce finite fit-quality metrics")
    return model


def summarize_fit_quality(metrics: pd.DataFrame) -> pd.DataFrame:
    """Summarize participant-level fits by channel."""
    rows = []
    for channel, group in metrics.groupby("channel", sort=True):
        r_squared = group["fit_r_squared"].to_numpy(float)
        error = group["fit_mae_log10_power"].to_numpy(float)
        rows.append(
            {
                "channel": channel,
                "n_participants": len(group),
                "n_trials": int(group["n_trials"].sum()),
                "median_r_squared": np.median(r_squared),
                "iqr_r_squared": np.subtract(*np.percentile(r_squared, [75, 25])),
                "median_mae_log10_power": np.median(error),
                "iqr_mae_log10_power": np.subtract(*np.percentile(error, [75, 25])),
            }
        )
    return pd.DataFrame(rows)


def analyze(args: argparse.Namespace) -> dict[str, pd.DataFrame]:
    participants = load_participants(args.participants)
    if participants.empty:
        raise ValueError("The participant table is empty")
    exclusions = load_exclusions(args.exclusions)
    frequencies = np.fft.rfftfreq(N_SAMPLES, d=1.0 / SAMPLE_RATE_HZ)
    fit_mask = (frequencies >= FIT_RANGE_HZ[0]) & (frequencies <= FIT_RANGE_HZ[1])
    metric_rows = []
    participant_spectra = []
    grand_spectra = []

    for channel in CHANNELS:
        participant_means = []
        channel_trial_count = 0
        for participant in participants.itertuples(index=False):
            power_sum = np.zeros(frequencies.size)
            trial_count = 0
            for number in range(1, 101):
                if (participant.participant_id, number) in exclusions:
                    continue
                day = "day1" if number <= 50 else "day2"
                path = find_processed_trial(
                    args.processed_root, participant.directory, day, channel, number
                )
                waveform = read_waveform(path, dtype="float64")
                # Average single-trial power spectra within each participant.
                power_sum += trial_power_spectrum(waveform)
                trial_count += 1
            if trial_count == 0:
                raise ValueError(f"No retained trials for {participant.participant_id}")

            mean_power = power_sum / trial_count
            model = fit_spectrum(frequencies, mean_power)
            participant_means.append(mean_power)
            channel_trial_count += trial_count
            metric_rows.append(
                {
                    "participant_id": participant.participant_id,
                    "channel": channel,
                    "n_trials": trial_count,
                    "aperiodic_offset": float(model.aperiodic_params_[0]),
                    "aperiodic_exponent": float(model.aperiodic_params_[1]),
                    "fit_r_squared": float(model.r_squared_),
                    # FOOOF's default error is MAE in log10-power space.
                    "fit_mae_log10_power": float(model.error_),
                }
            )
            participant_spectra.append(
                pd.DataFrame(
                    {
                        "participant_id": participant.participant_id,
                        "channel": channel,
                        "frequency_hz": frequencies[fit_mask],
                        "mean_trial_power_v2": mean_power[fit_mask],
                        "fitted_log10_power": model.fooofed_spectrum_,
                    }
                )
            )

        # Fit the equally weighted mean of participant spectra.
        grand_mean = np.mean(np.vstack(participant_means), axis=0)
        grand_model = fit_spectrum(frequencies, grand_mean)
        grand_spectra.append(
            pd.DataFrame(
                {
                    "channel": channel,
                    "n_participants": len(participant_means),
                    "n_trials": channel_trial_count,
                    "frequency_hz": frequencies[fit_mask],
                    "mean_participant_power_v2": grand_mean[fit_mask],
                    "fitted_log10_power": grand_model.fooofed_spectrum_,
                }
            )
        )

    metrics = pd.DataFrame(metric_rows).sort_values(["channel", "participant_id"])
    return {
        "participant_fit_metrics.csv": metrics,
        "fit_quality_summary.csv": summarize_fit_quality(metrics),
        "participant_spectra.csv": pd.concat(participant_spectra, ignore_index=True),
        "grand_average_spectra.csv": pd.concat(grand_spectra, ignore_index=True),
    }


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
        "--results-dir", type=Path, default=PROJECT_ROOT / "results/spectral_summary"
    )
    args = parser.parse_args()
    tables = analyze(args)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    for filename, frame in tables.items():
        frame.to_csv(args.results_dir / filename, index=False)


if __name__ == "__main__":
    main()

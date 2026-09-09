"""Create 2-second pre-movement EEG segments from the raw recordings."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, iirnotch, lfilter

from common import (
    CHANNELS,
    N_SAMPLES,
    PROJECT_ROOT,
    SAMPLE_RATE_HZ,
    load_participants,
    parse_trial_number,
)


CHANNEL_COLUMNS = {"c3": "EEG_C3", "cz": "EEG_Cz", "c4": "EEG_C4"}


def acceleration_reference(acceleration: np.ndarray) -> float:
    """Average the first 100 non-zero acceleration observations."""
    total = 0.0
    nonzero_count = 0
    for value in np.nan_to_num(acceleration, nan=0.0):
        if nonzero_count >= 100:
            break
        total += float(value)
        if value != 0:
            nonzero_count += 1
    return total / 100.0


def movement_onset(acceleration: np.ndarray) -> tuple[int, float, str]:
    """Find the first change greater than 2 m/s2, or raise if none is detected."""
    clean = np.nan_to_num(acceleration, nan=0.0)
    reference = acceleration_reference(clean)
    for index in range(N_SAMPLES, clean.size):
        value = clean[index]
        if value != 0 and abs(float(value) - reference) > 2.0:
            return index, reference, "threshold"

    raise ValueError(
        "Movement onset not detected: no non-zero acceleration value after "
        f"the first {N_SAMPLES} samples differs from the baseline by more "
        "than 2 m/s2."
    )


def filter_eeg(values: np.ndarray) -> np.ndarray:
    """Apply causal 50-Hz notch and 3-70-Hz band-pass filters."""
    notch_b, notch_a = iirnotch(50.0 / (0.5 * SAMPLE_RATE_HZ), 30.0)
    band_b, band_a = butter(
        3,
        [3.0 / (0.5 * SAMPLE_RATE_HZ), 70.0 / (0.5 * SAMPLE_RATE_HZ)],
        btype="band",
    )
    return lfilter(band_b, band_a, lfilter(notch_b, notch_a, values))


def preprocess_trial(raw_path: Path) -> tuple[dict[str, np.ndarray], dict]:
    frame = pd.read_csv(raw_path)
    required = {"ACC_TOP", *CHANNEL_COLUMNS.values()}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{raw_path} is missing columns {sorted(missing)}")

    onset, reference, onset_source = movement_onset(
        frame["ACC_TOP"].to_numpy(float)
    )
    start = onset - N_SAMPLES
    if start < 0:
        raise ValueError(f"Insufficient pre-movement samples in {raw_path}")

    waveforms = {}
    for channel, column in CHANNEL_COLUMNS.items():
        filtered = filter_eeg(frame[column].to_numpy(float)[:onset])
        waveform = np.asarray(filtered[start:onset], dtype=float)
        if waveform.size != N_SAMPLES:
            raise ValueError(f"Unexpected segment length in {raw_path}")
        waveforms[channel] = waveform

    return waveforms, {
        "movement_onset_sample": onset,
        "segment_start_sample": start,
        "acceleration_reference_m_s2": reference,
        "movement_onset_source": onset_source,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=PROJECT_ROOT / "data/raw")
    parser.add_argument(
        "--processed-root", type=Path, default=PROJECT_ROOT / "data/processed"
    )
    parser.add_argument(
        "--participants",
        type=Path,
        default=PROJECT_ROOT / "data/participants.csv",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT / "results/preprocessing.csv",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    manifest = []
    for participant in load_participants(args.participants).itertuples(index=False):
        raw_folder = args.raw_root / participant.directory
        raw_files = sorted(raw_folder.glob("*.csv"), key=parse_trial_number)
        trial_numbers = [parse_trial_number(path) for path in raw_files]
        if trial_numbers != list(range(1, 101)):
            raise ValueError(f"Expected trials 1-100 in {raw_folder}")

        for raw_path, number in zip(raw_files, trial_numbers):
            waveforms, metadata = preprocess_trial(raw_path)
            day = "day1" if number <= 50 else "day2"
            output_folder = (
                args.processed_root / participant.directory / f"{day}_all"
            )
            output_folder.mkdir(parents=True, exist_ok=True)

            # All three channels are written in one pass.
            for channel in CHANNELS:
                output_path = output_folder / (
                    f"{participant.participant_id}_filtered_"
                    f"{channel}_{number:03d}.csv"
                )
                if output_path.exists() and not args.overwrite:
                    raise FileExistsError(
                        f"{output_path} already exists; use --overwrite to replace it"
                    )
                pd.DataFrame(waveforms[channel]).to_csv(
                    output_path, header=False, index=False
                )

            manifest.append(
                {
                    "participant_id": participant.participant_id,
                    "trial_number": number,
                    "day": day,
                    "raw_file": raw_path.name,
                    **metadata,
                }
            )

    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(manifest).to_csv(args.manifest, index=False)


if __name__ == "__main__":
    main()

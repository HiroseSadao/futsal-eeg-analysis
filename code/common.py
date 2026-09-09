"""Shared constants and input helpers for the analysis scripts."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

SAMPLE_RATE_HZ = 200.0
N_SAMPLES = 400
CHANNELS = ("c3", "cz", "c4")
DAYS = ("day1", "day2")
DAY_TRIALS = {
    "day1": range(1, 51),
    "day2": range(51, 101),
}
RANDOM_SEED = 42

TRIAL_NUMBER_PATTERN = re.compile(r"_(\d+)\.csv$", re.IGNORECASE)


def parse_trial_number(path: str | Path) -> int:
    """Read the final integer before the .csv extension."""
    match = TRIAL_NUMBER_PATTERN.search(Path(path).name)
    if match is None:
        raise ValueError(f"Cannot identify the trial number in {path}")
    return int(match.group(1))


def load_participants(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype=str)
    required = {"participant_id", "directory"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing participant columns: {sorted(missing)}")
    if frame["participant_id"].duplicated().any():
        raise ValueError("participant_id values must be unique")
    return frame.loc[:, ["participant_id", "directory"]]


def load_outcomes(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"participant_id", "trial_number", "day", "outcome"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing outcome columns: {sorted(missing)}")
    frame["participant_id"] = frame["participant_id"].astype(str)
    frame["trial_number"] = frame["trial_number"].astype(int)
    frame["outcome"] = frame["outcome"].astype(int)
    if not set(frame["day"]).issubset(DAYS):
        raise ValueError("The outcome table contains an invalid day")
    if not set(frame["outcome"]).issubset({0, 1}):
        raise ValueError("Outcome must be 0 (failure) or 1 (success)")
    if frame.duplicated(["participant_id", "trial_number"]).any():
        raise ValueError("Outcomes must be unique by participant and trial")
    return frame


def load_exclusions(path: Path) -> set[tuple[str, int]]:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} does not exist. Run 02_fooof_exclusion.py first."
        )
    frame = pd.read_csv(path)
    required = {"participant_id", "trial_number"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing exclusion columns: {sorted(missing)}")
    return {
        (str(row.participant_id), int(row.trial_number))
        for row in frame.itertuples(index=False)
    }


def find_processed_trial(
    processed_root: Path,
    directory: str,
    day: str,
    channel: str,
    trial_number: int,
) -> Path:
    folder = processed_root / directory / f"{day}_all"
    matches = sorted(folder.glob(f"*_filtered_{channel}_{trial_number:03d}.csv"))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected one file for {directory}/{day}/{channel}/"
            f"trial {trial_number:03d}; found {len(matches)}"
        )
    return matches[0]


def read_waveform(path: Path, dtype: str = "float32") -> np.ndarray:
    waveform = pd.read_csv(path, header=None).iloc[:, 0].to_numpy(dtype=dtype)
    if waveform.size != N_SAMPLES:
        raise ValueError(f"Expected {N_SAMPLES} samples in {path}")
    if not np.isfinite(waveform).all():
        raise ValueError(f"Non-finite values found in {path}")
    return waveform


def load_task(
    processed_root: Path,
    directory: str,
    participant_id: str,
    day: str,
    channel: str,
    outcomes: pd.DataFrame,
    exclusions: set[tuple[str, int]],
    trial_order: list[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load trials in the recorded analysis order, then apply exclusions."""
    if sorted(trial_order) != list(DAY_TRIALS[day]):
        raise ValueError(f"Invalid trial order for {participant_id}/{day}/{channel}")
    participant_outcomes = outcomes.loc[
        outcomes["participant_id"] == participant_id
    ]
    labels_by_trial = dict(
        zip(
            participant_outcomes["trial_number"].astype(int),
            participant_outcomes["outcome"].astype(int),
        )
    )

    waveforms = []
    labels = []
    trial_numbers = []
    for number in trial_order:
        if (participant_id, number) in exclusions:
            continue
        if number not in labels_by_trial:
            raise KeyError(f"Missing outcome for {participant_id}, trial {number}")
        path = find_processed_trial(
            processed_root, directory, day, channel, number
        )
        waveforms.append(read_waveform(path))
        labels.append(labels_by_trial[number])
        trial_numbers.append(number)

    features = np.vstack(waveforms).astype("float32")
    label_array = np.asarray(labels, dtype="int64")
    if np.unique(label_array).size != 2:
        raise ValueError(f"Both classes are required for {participant_id}/{day}")
    return (
        features,
        label_array,
        np.asarray(trial_numbers, dtype="int64"),
    )


def stable_seed(*parts: object) -> int:
    key = "|".join(str(part) for part in parts)
    digest = hashlib.blake2b(key.encode("utf-8"), digest_size=4).digest()
    return (RANDOM_SEED + int.from_bytes(digest, "little")) % (2**32 - 1)


def empirical_upper_p(observed: float, null_values: np.ndarray) -> float:
    null_array = np.asarray(null_values, dtype="float64")
    null_array = null_array[np.isfinite(null_array)]
    if not np.isfinite(observed) or null_array.size == 0:
        return float("nan")
    return float((1 + np.sum(null_array >= observed)) / (null_array.size + 1))


def holm_adjust(p_values: np.ndarray) -> np.ndarray:
    """Holm-adjust p-values and return them in their original order."""
    p_array = np.asarray(p_values, dtype="float64")
    adjusted = np.full_like(p_array, np.nan)
    valid = np.flatnonzero(np.isfinite(p_array))
    ordered = valid[np.argsort(p_array[valid])]
    running_max = 0.0
    for rank, original_index in enumerate(ordered):
        candidate = min(1.0, (ordered.size - rank) * p_array[original_index])
        running_max = max(running_max, candidate)
        adjusted[original_index] = running_max
    return adjusted

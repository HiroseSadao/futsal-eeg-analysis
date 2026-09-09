"""Run the within-day and cross-day linear SVM analyses."""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import (
    auc,
    f1_score,
    hinge_loss,
    precision_score,
    recall_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from common import (
    CHANNELS,
    DAYS,
    PROJECT_ROOT,
    RANDOM_SEED,
    empirical_upper_p,
    holm_adjust,
    load_exclusions,
    load_outcomes,
    load_participants,
    load_task,
)
from svm_design import load_design


C_VALUES = (0.01, 0.1, 1.0, 10.0, 100.0)
N_FOLDS = 4


def compute_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    false_positive_rate, true_positive_rate, _ = roc_curve(labels, scores)
    return float(auc(false_positive_rate, true_positive_rate))


def classifier(c_value: float, seed: int) -> Pipeline:
    # Standardization is learned from each training fold only.
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "svm",
                LinearSVC(
                    C=c_value,
                    class_weight="balanced",
                    dual="auto",
                    max_iter=10_000,
                    random_state=seed,
                ),
            ),
        ]
    )


def evaluate_cv(
    features: np.ndarray,
    labels: np.ndarray,
    c_value: float,
    base_seed: int,
) -> tuple[dict[str, float], list[dict]]:
    splitter = StratifiedKFold(
        n_splits=N_FOLDS,
        shuffle=True,
        random_state=base_seed,
    )
    fold_rows = []
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        for fold, (train_indices, validation_indices) in enumerate(
            splitter.split(features, labels), start=1
        ):
            model = classifier(c_value, base_seed + fold)
            model.fit(features[train_indices], labels[train_indices])
            validation_labels = labels[validation_indices]
            predictions = model.predict(features[validation_indices])
            scores = model.decision_function(features[validation_indices])
            fold_rows.append(
                {
                    "fold": fold,
                    "hinge_loss": hinge_loss(validation_labels, scores),
                    "macro_precision": precision_score(
                        validation_labels,
                        predictions,
                        average="macro",
                        zero_division=0,
                    ),
                    "macro_recall": recall_score(
                        validation_labels,
                        predictions,
                        average="macro",
                        zero_division=0,
                    ),
                    "macro_f1": f1_score(
                        validation_labels,
                        predictions,
                        average="macro",
                        zero_division=0,
                    ),
                    "roc_auc": compute_auc(validation_labels, scores),
                }
            )

    folds = pd.DataFrame(fold_rows)
    return {
        "c_value": c_value,
        "mean_hinge_loss": folds["hinge_loss"].mean(),
        "mean_macro_precision": folds["macro_precision"].mean(),
        "mean_macro_recall": folds["macro_recall"].mean(),
        "mean_macro_f1": folds["macro_f1"].mean(),
        "mean_roc_auc": folds["roc_auc"].mean(),
    }, fold_rows


def c_seed(c_value: float) -> int:
    for index, candidate in enumerate(C_VALUES):
        if np.isclose(c_value, candidate):
            return RANDOM_SEED + index * 100
    raise ValueError(f"C={c_value} is not in the parameter grid")


def select_c(
    features: np.ndarray,
    labels: np.ndarray,
) -> tuple[dict[str, float], list[dict]]:
    best = None
    best_folds = None
    for index, c_value in enumerate(C_VALUES):
        result, folds = evaluate_cv(
            features, labels, c_value, RANDOM_SEED + index * 100
        )
        is_better = best is None or (
            result["mean_macro_f1"] > best["mean_macro_f1"]
            or (
                np.isclose(result["mean_macro_f1"], best["mean_macro_f1"])
                and result["mean_hinge_loss"] < best["mean_hinge_loss"]
            )
        )
        if is_better:
            best = result
            best_folds = folds
    return best, best_folds


def load_all_tasks(args: argparse.Namespace, design: dict) -> dict:
    participants = load_participants(args.participants)
    outcomes = load_outcomes(args.outcomes)
    exclusions = load_exclusions(args.exclusions)
    expected = {
        (participant_id, day, channel)
        for participant_id in participants["participant_id"]
        for day in DAYS
        for channel in CHANNELS
    }
    if set(design) != expected:
        raise ValueError(
            "SVM design tasks must match all participant/day/channel combinations"
        )
    tasks = {}
    for participant in participants.itertuples(index=False):
        for day in DAYS:
            for channel in CHANNELS:
                tasks[(participant.participant_id, day, channel)] = load_task(
                    args.processed_root,
                    participant.directory,
                    participant.participant_id,
                    day,
                    channel,
                    outcomes,
                    exclusions,
                    design[(participant.participant_id, day, channel)]["trial_order"],
                )
    return tasks


def within_day(tasks: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    task_rows = []
    fold_rows = []
    for (participant_id, day, channel), (
        features,
        labels,
        trial_numbers,
    ) in sorted(tasks.items()):
        result, folds = select_c(features, labels)
        task_rows.append(
            {
                "participant_id": participant_id,
                "day": day,
                "channel": channel,
                "n_trials": labels.size,
                "n_success": np.sum(labels == 1),
                "n_failure": np.sum(labels == 0),
                "first_trial_in_analysis_order": trial_numbers[0],
                **result,
            }
        )
        splitter = StratifiedKFold(
            n_splits=N_FOLDS, shuffle=True, random_state=c_seed(result["c_value"])
        )
        for fold, (train_indices, validation_indices) in zip(
            folds, splitter.split(features, labels)
        ):
            fold_rows.append(
                {
                    "participant_id": participant_id,
                    "day": day,
                    "channel": channel,
                    "c_value": result["c_value"],
                    **fold,
                    "training_trials": ";".join(
                        map(str, trial_numbers[train_indices])
                    ),
                    "validation_trials": ";".join(
                        map(str, trial_numbers[validation_indices])
                    ),
                }
            )
    return pd.DataFrame(task_rows), pd.DataFrame(fold_rows)


def within_day_permutations(
    tasks: dict,
    within_results: pd.DataFrame,
    n_permutations: int,
    design: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected_c = {
        (row.participant_id, row.day, row.channel): float(row.c_value)
        for row in within_results.itertuples(index=False)
    }
    task_rows = []
    null_rows = []

    for key, (features, labels, _) in sorted(tasks.items()):
        participant_id, day, channel = key
        c_value = selected_c[key]
        observed, _ = evaluate_cv(features, labels, c_value, c_seed(c_value))
        task_null_rows = []

        for permutation_index in range(n_permutations):
            # Use the recorded permutation seed.
            seed = design[key]["within_seeds"][permutation_index]
            shuffled = np.random.RandomState(seed).permutation(labels)
            null_result, _ = evaluate_cv(
                features, shuffled, c_value, c_seed(c_value)
            )
            row = {
                "participant_id": participant_id,
                "day": day,
                "channel": channel,
                "permutation_index": permutation_index,
                "seed": seed,
                "null_auc": null_result["mean_roc_auc"],
                "null_f1": null_result["mean_macro_f1"],
            }
            task_null_rows.append(row)
            null_rows.append(row)

        null_auc = np.asarray([row["null_auc"] for row in task_null_rows])
        null_f1 = np.asarray([row["null_f1"] for row in task_null_rows])
        task_rows.append(
            {
                "participant_id": participant_id,
                "day": day,
                "channel": channel,
                "n_trials": labels.size,
                "observed_c_value": c_value,
                "observed_auc": observed["mean_roc_auc"],
                "observed_f1": observed["mean_macro_f1"],
                "null_auc_mean": np.mean(null_auc),
                "null_auc_std": np.std(null_auc),
                "p_auc": empirical_upper_p(
                    observed["mean_roc_auc"], null_auc
                ),
                "null_f1_mean": np.mean(null_f1),
                "null_f1_std": np.std(null_f1),
                "p_f1": empirical_upper_p(
                    observed["mean_macro_f1"], null_f1
                ),
            }
        )
    return pd.DataFrame(task_rows), pd.DataFrame(null_rows)


def evaluate_cross_day(
    source_features: np.ndarray,
    source_labels: np.ndarray,
    target_features: np.ndarray,
    target_labels: np.ndarray,
    c_value: float,
) -> dict:
    base_seed = c_seed(c_value)
    splitter = StratifiedKFold(
        n_splits=N_FOLDS, shuffle=True, random_state=base_seed
    )
    source_rows = []
    target_fold_auc = []
    target_scores = []

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        for fold, (train_indices, validation_indices) in enumerate(
            splitter.split(source_features, source_labels), start=1
        ):
            model = classifier(c_value, base_seed + fold)
            model.fit(source_features[train_indices], source_labels[train_indices])
            validation_labels = source_labels[validation_indices]
            validation_scores = model.decision_function(
                source_features[validation_indices]
            )
            validation_predictions = model.predict(
                source_features[validation_indices]
            )
            fold_target_scores = model.decision_function(target_features)
            source_rows.append(
                {
                    "hinge_loss": hinge_loss(
                        validation_labels, validation_scores
                    ),
                    "macro_f1": f1_score(
                        validation_labels,
                        validation_predictions,
                        average="macro",
                        zero_division=0,
                    ),
                    "roc_auc": compute_auc(
                        validation_labels, validation_scores
                    ),
                }
            )
            target_fold_auc.append(
                compute_auc(target_labels, fold_target_scores)
            )
            target_scores.append(fold_target_scores)

    source = pd.DataFrame(source_rows)
    ensemble_scores = np.mean(np.vstack(target_scores), axis=0)
    predictions = (ensemble_scores >= 0).astype("int64")
    return {
        "source_cv_auc": source["roc_auc"].mean(),
        "source_cv_f1": source["macro_f1"].mean(),
        "source_cv_loss": source["hinge_loss"].mean(),
        "crossday_auc": compute_auc(target_labels, ensemble_scores),
        "crossday_f1": f1_score(
            target_labels, predictions, average="macro", zero_division=0
        ),
        "crossday_precision": precision_score(
            target_labels, predictions, average="macro", zero_division=0
        ),
        "crossday_recall": recall_score(
            target_labels, predictions, average="macro", zero_division=0
        ),
        "fold_target_auc_mean": np.mean(target_fold_auc),
        "fold_target_auc_std": np.std(target_fold_auc),
        "target_scores": ensemble_scores,
    }


def cross_day(
    tasks: dict,
    within_results: pd.DataFrame,
    n_permutations: int,
    design: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected_c = {
        (row.participant_id, row.day, row.channel): float(row.c_value)
        for row in within_results.itertuples(index=False)
    }
    participant_ids = sorted({key[0] for key in tasks})
    task_rows = []
    null_rows = []

    for participant_id in participant_ids:
        for channel in CHANNELS:
            for train_day, test_day in (("day1", "day2"), ("day2", "day1")):
                source_features, source_labels, _ = tasks[
                    (participant_id, train_day, channel)
                ]
                target_features, target_labels, _ = tasks[
                    (participant_id, test_day, channel)
                ]
                c_value = selected_c[(participant_id, train_day, channel)]
                observed = evaluate_cross_day(
                    source_features,
                    source_labels,
                    target_features,
                    target_labels,
                    c_value,
                )
                predictions = (observed["target_scores"] >= 0).astype("int64")
                task_null_rows = []

                for permutation_index in range(n_permutations):
                    seed = design[(participant_id, train_day, channel)]["cross_seeds"][
                        permutation_index
                    ]
                    shuffled = np.random.RandomState(seed).permutation(
                        target_labels
                    )
                    row = {
                        "participant_id": participant_id,
                        "channel": channel,
                        "train_day": train_day,
                        "test_day": test_day,
                        "permutation_index": permutation_index,
                        "seed": seed,
                        "null_auc": compute_auc(
                            shuffled, observed["target_scores"]
                        ),
                        "null_f1": f1_score(
                            shuffled,
                            predictions,
                            average="macro",
                            zero_division=0,
                        ),
                    }
                    task_null_rows.append(row)
                    null_rows.append(row)

                null_auc = np.asarray(
                    [row["null_auc"] for row in task_null_rows]
                )
                null_f1 = np.asarray(
                    [row["null_f1"] for row in task_null_rows]
                )
                task_rows.append(
                    {
                        "participant_id": participant_id,
                        "channel": channel,
                        "train_day": train_day,
                        "test_day": test_day,
                        "n_source_trials": source_labels.size,
                        "n_target_trials": target_labels.size,
                        "observed_c_value": c_value,
                        "observed_auc": observed["crossday_auc"],
                        "observed_f1": observed["crossday_f1"],
                        "observed_precision": observed["crossday_precision"],
                        "observed_recall": observed["crossday_recall"],
                        "fold_target_auc_mean": observed[
                            "fold_target_auc_mean"
                        ],
                        "fold_target_auc_std": observed[
                            "fold_target_auc_std"
                        ],
                        "source_cv_auc": observed["source_cv_auc"],
                        "source_cv_f1": observed["source_cv_f1"],
                        "source_cv_loss": observed["source_cv_loss"],
                        "null_auc_mean": np.mean(null_auc),
                        "null_auc_std": np.std(null_auc),
                        "p_auc": empirical_upper_p(
                            observed["crossday_auc"], null_auc
                        ),
                        "null_f1_mean": np.mean(null_f1),
                        "null_f1_std": np.std(null_f1),
                        "p_f1": empirical_upper_p(
                            observed["crossday_f1"], null_f1
                        ),
                    }
                )
    return pd.DataFrame(task_rows), pd.DataFrame(null_rows)


def channel_summary(
    task_results: pd.DataFrame,
    null_results: pd.DataFrame,
    task_count_name: str,
) -> pd.DataFrame:
    """Test the group mean and apply Holm correction across channels."""
    rows = []
    for metric in ("auc", "f1"):
        metric_row_indices = []
        for channel in CHANNELS:
            tasks = task_results.loc[task_results["channel"] == channel]
            nulls = null_results.loc[null_results["channel"] == channel]
            observed_subject_mean = (
                tasks.groupby("participant_id")[f"observed_{metric}"]
                .mean()
                .mean()
            )
            subject_null_means = (
                nulls.groupby(["participant_id", "permutation_index"])[
                    f"null_{metric}"
                ]
                .mean()
                .groupby("permutation_index")
                .mean()
                .to_numpy(float)
            )
            rows.append(
                {
                    "channel": channel,
                    "metric": metric,
                    task_count_name: tasks.shape[0],
                    "n_participants": tasks["participant_id"].nunique(),
                    "observed_subject_mean": observed_subject_mean,
                    "null_subject_mean": np.mean(subject_null_means),
                    "null_subject_std": np.std(subject_null_means),
                    "p_subject_mean": empirical_upper_p(
                        observed_subject_mean, subject_null_means
                    ),
                    "holm_p_subject_mean": np.nan,
                }
            )
            metric_row_indices.append(len(rows) - 1)

        adjusted = holm_adjust(
            np.asarray(
                [rows[index]["p_subject_mean"] for index in metric_row_indices]
            )
        )
        for index, value in zip(metric_row_indices, adjusted):
            rows[index]["holm_p_subject_mean"] = value
    return pd.DataFrame(rows).sort_values(["metric", "channel"])


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
        "--results-dir", type=Path, default=PROJECT_ROOT / "results/svm"
    )
    parser.add_argument(
        "--design",
        type=Path,
        default=PROJECT_ROOT / "data/svm_design.json",
        help="Fixed trial order and permutation seeds recorded before file reorganization",
    )
    parser.add_argument("--permutations", type=int, default=199)
    args = parser.parse_args()

    design = load_design(args.design, args.permutations)
    tasks = load_all_tasks(args, design)
    within_tasks, within_folds = within_day(tasks)
    within_permutation_tasks, within_null = within_day_permutations(
        tasks, within_tasks, args.permutations, design
    )
    within_summary = channel_summary(
        within_permutation_tasks, within_null, "n_participant_days"
    )
    cross_tasks, cross_null = cross_day(
        tasks, within_tasks, args.permutations, design
    )
    cross_summary = channel_summary(
        cross_tasks, cross_null, "n_directions"
    )

    args.results_dir.mkdir(parents=True, exist_ok=True)
    tables = {
        "within_day_tasks.csv": within_tasks,
        "within_day_folds.csv": within_folds,
        "within_day_permutation_tasks.csv": within_permutation_tasks,
        "within_day_permutation_null.csv": within_null,
        "within_day_summary.csv": within_summary,
        "cross_day_tasks.csv": cross_tasks,
        "cross_day_null.csv": cross_null,
        "cross_day_summary.csv": cross_summary,
    }
    for filename, frame in tables.items():
        frame.to_csv(args.results_dir / filename, index=False)


if __name__ == "__main__":
    main()

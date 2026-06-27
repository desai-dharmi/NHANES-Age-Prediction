"""Threshold optimization and submission generation for NHANES Age Prediction."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from project_setup import (
    Config,
    ensure_directories,
    print_section,
    save_dataframe,
    set_random_seed,
    setup_logging,
    timer,
)

logger = logging.getLogger("nhanes")


@dataclass
class ThresholdResult:
    """Container for threshold search outputs."""

    model_name: str
    threshold: float
    oof_f1: float
    oof_predictions: np.ndarray
    test_predictions: np.ndarray


def load_model_comparison(path: Path) -> pd.DataFrame:
    """Load ranked model comparison results."""
    if not path.is_file():
        raise FileNotFoundError(f"Model comparison file not found: {path}")
    return pd.read_csv(path)


def load_probability_file(path: Path) -> pd.DataFrame:
    """Load OOF or test probability outputs."""
    if not path.is_file():
        raise FileNotFoundError(f"Probability file not found: {path}")
    return pd.read_csv(path)


def select_best_model(comparison_df: pd.DataFrame) -> str:
    """Select the top-ranked model from comparison results."""
    if comparison_df.empty:
        raise ValueError("Model comparison dataframe is empty")
    if "model" not in comparison_df.columns:
        raise ValueError("Model comparison dataframe missing 'model' column")
    return str(comparison_df.iloc[0]["model"])


def optimize_threshold(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    thresholds: np.ndarray | None = None,
) -> tuple[float, float]:
    """Find the probability threshold that maximizes F1 score."""
    if thresholds is None:
        thresholds = np.linspace(0.05, 0.95, 181)
    best_threshold = 0.5
    best_f1 = -1.0
    for threshold in thresholds:
        predictions = (probabilities >= threshold).astype(int)
        score = float(f1_score(y_true, predictions, zero_division=0))
        if score > best_f1:
            best_f1 = score
            best_threshold = float(threshold)
    return best_threshold, best_f1


def apply_threshold(probabilities: np.ndarray, threshold: float) -> np.ndarray:
    """Convert probabilities to binary predictions using a threshold."""
    return (probabilities >= threshold).astype(int)


def build_submission(predictions: np.ndarray) -> pd.DataFrame:
    """Build a competition submission dataframe."""
    return pd.DataFrame({"age_group": predictions.astype(int)})


def generate_predictions(
    model_name: str,
    config: Config,
    thresholds: np.ndarray | None = None,
) -> ThresholdResult:
    """Optimize threshold on OOF data and generate test predictions."""
    oof_path = config.output_dir / f"{model_name}_oof.csv"
    test_path = config.output_dir / f"{model_name}_test.csv"

    oof_df = load_probability_file(oof_path)
    test_df = load_probability_file(test_path)

    required_columns = {"y_true", "proba_senior"}
    missing = required_columns - set(oof_df.columns)
    if missing:
        raise ValueError(f"OOF file for {model_name} missing columns: {sorted(missing)}")
    if "proba_senior" not in test_df.columns:
        raise ValueError(f"Test probability file for {model_name} missing 'proba_senior'")

    y_true = oof_df["y_true"].to_numpy(dtype=int)
    oof_probabilities = oof_df["proba_senior"].to_numpy(dtype=float)
    test_probabilities = test_df["proba_senior"].to_numpy(dtype=float)

    threshold, oof_f1 = optimize_threshold(
        y_true,
        oof_probabilities,
        thresholds=thresholds,
    )
    oof_predictions = apply_threshold(oof_probabilities, threshold)
    test_predictions = apply_threshold(test_probabilities, threshold)

    return ThresholdResult(
        model_name=model_name,
        threshold=threshold,
        oof_f1=oof_f1,
        oof_predictions=oof_predictions,
        test_predictions=test_predictions,
    )


def save_threshold_report(result: ThresholdResult, path: Path) -> Path:
    """Save threshold optimization results to CSV."""
    report = pd.DataFrame(
        [
            {
                "model": result.model_name,
                "threshold": result.threshold,
                "oof_f1": result.oof_f1,
                "test_positive_rate": float(result.test_predictions.mean()),
            }
        ]
    )
    return save_dataframe(report, path)


@timer
def main() -> None:
    """Generate an optimized submission from cross-validated model outputs."""
    config = Config()
    ensure_directories(config)
    setup_logging(config)
    set_random_seed(config.random_seed)

    print_section("Loading model comparison...")
    comparison_path = config.output_dir / "model_comparison.csv"
    comparison_df = load_model_comparison(comparison_path)
    best_model = select_best_model(comparison_df)
    logger.info("Selected model: %s", best_model)

    print_section("Optimizing threshold...")
    result = generate_predictions(best_model, config)

    print_section("Saving submission...")
    submission = build_submission(result.test_predictions)
    submission_path = config.submissions_dir / "submission.csv"
    save_dataframe(submission, submission_path)

    threshold_path = config.output_dir / "threshold_report.csv"
    save_threshold_report(result, threshold_path)

    logger.info("Optimal threshold: %.4f", result.threshold)
    logger.info("OOF F1 at optimal threshold: %.4f", result.oof_f1)
    logger.info("Saved submission to %s", submission_path)
    print_section("Completed successfully.")


if __name__ == "__main__":
    import sys

    inference_module = ModuleType("inference")
    inference_module.__dict__.update(globals())
    sys.modules["inference"] = inference_module
    for _pickle_class in (ThresholdResult,):
        _pickle_class.__module__ = "inference"
    main()

"""Multi-model training and cross-validation for NHANES Age Prediction."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any, Callable

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

from project_setup import (
    Config,
    TARGET_LABEL_TO_INT,
    ensure_directories,
    print_section,
    set_random_seed,
    setup_logging,
    timer,
)

logger = logging.getLogger("nhanes")

ModelTrainer = Callable[..., Any]
MetricName = str

METRIC_NAMES: tuple[MetricName, ...] = (
    "f1",
    "precision",
    "recall",
    "roc_auc",
    "accuracy",
)


@dataclass
class CVResult:
    """Container for cross-validation outputs."""

    model_name: str
    oof_probabilities: np.ndarray
    oof_predictions: np.ndarray
    test_probabilities: np.ndarray
    fold_metrics: pd.DataFrame
    fold_models: list[Any] = field(default_factory=list)


def _merge_params(defaults: dict[str, Any], params: dict[str, Any] | None) -> dict[str, Any]:
    """Merge user parameters over model defaults."""
    merged = defaults.copy()
    if params:
        merged.update(params)
    return merged


def default_catboost_params(random_state: int) -> dict[str, Any]:
    """Return default CatBoost parameters."""
    return {
        "iterations": 1000,
        "learning_rate": 0.05,
        "depth": 6,
        "l2_leaf_reg": 5.0,
        "random_seed": random_state,
        "loss_function": "Logloss",
        "eval_metric": "F1",
        "auto_class_weights": "Balanced",
        "verbose": False,
    }


def default_lightgbm_params(random_state: int) -> dict[str, Any]:
    """Return default LightGBM parameters."""
    return {
        "n_estimators": 1000,
        "learning_rate": 0.05,
        "max_depth": 6,
        "num_leaves": 31,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_lambda": 1.0,
        "class_weight": "balanced",
        "random_state": random_state,
        "verbose": -1,
    }


def default_xgboost_params(random_state: int, scale_pos_weight: float) -> dict[str, Any]:
    """Return default XGBoost parameters."""
    return {
        "n_estimators": 1000,
        "learning_rate": 0.05,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_lambda": 1.0,
        "scale_pos_weight": scale_pos_weight,
        "random_state": random_state,
        "eval_metric": "logloss",
        "verbosity": 0,
        "n_jobs": -1,
    }


def default_extratrees_params(random_state: int) -> dict[str, Any]:
    """Return default ExtraTrees parameters."""
    return {
        "n_estimators": 500,
        "max_depth": None,
        "min_samples_leaf": 2,
        "max_features": "sqrt",
        "class_weight": "balanced",
        "random_state": random_state,
        "n_jobs": -1,
    }


def train_catboost(
    X_train: pd.DataFrame,
    y_train: pd.Series | np.ndarray,
    params: dict[str, Any] | None = None,
    random_state: int = 42,
) -> CatBoostClassifier:
    """Train a CatBoost classifier."""
    model_params = _merge_params(default_catboost_params(random_state), params)
    model = CatBoostClassifier(**model_params)
    model.fit(X_train, y_train)
    return model


def train_lightgbm(
    X_train: pd.DataFrame,
    y_train: pd.Series | np.ndarray,
    params: dict[str, Any] | None = None,
    random_state: int = 42,
) -> LGBMClassifier:
    """Train a LightGBM classifier."""
    model_params = _merge_params(default_lightgbm_params(random_state), params)
    model = LGBMClassifier(**model_params)
    model.fit(X_train, y_train)
    return model


def train_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series | np.ndarray,
    params: dict[str, Any] | None = None,
    random_state: int = 42,
    scale_pos_weight: float = 1.0,
) -> XGBClassifier:
    """Train an XGBoost classifier."""
    defaults = default_xgboost_params(random_state, scale_pos_weight)
    model_params = _merge_params(defaults, params)
    model = XGBClassifier(**model_params)
    model.fit(X_train, y_train)
    return model


def train_extratrees(
    X_train: pd.DataFrame,
    y_train: pd.Series | np.ndarray,
    params: dict[str, Any] | None = None,
    random_state: int = 42,
) -> ExtraTreesClassifier:
    """Train an ExtraTrees classifier."""
    model_params = _merge_params(default_extratrees_params(random_state), params)
    model = ExtraTreesClassifier(**model_params)
    model.fit(X_train, y_train)
    return model


def predict_proba(model: Any, X: pd.DataFrame) -> np.ndarray:
    """Return positive-class probabilities for a fitted model."""
    probabilities = model.predict_proba(X)
    if probabilities.ndim != 2 or probabilities.shape[1] < 2:
        raise ValueError("Model predict_proba output must contain two classes")
    return probabilities[:, 1]


def predict(model: Any, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
    """Return binary predictions using a probability threshold."""
    probabilities = predict_proba(model, X)
    return (probabilities >= threshold).astype(int)


def compute_fold_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
) -> dict[str, float]:
    """Compute classification metrics for one validation fold."""
    return {
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
    }


def create_stratified_folds(
    y: pd.Series | np.ndarray,
    n_folds: int,
    random_state: int,
    shuffle: bool = True,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Create reusable stratified cross-validation splits."""
    splitter = StratifiedKFold(
        n_splits=n_folds,
        shuffle=shuffle,
        random_state=random_state,
    )
    y_array = np.asarray(y)
    return list(splitter.split(np.zeros(len(y_array)), y_array))


def cross_validate_model(
    trainer: ModelTrainer,
    model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series | np.ndarray,
    X_test: pd.DataFrame,
    cv_splits: list[tuple[np.ndarray, np.ndarray]],
    trainer_kwargs: dict[str, Any] | None = None,
) -> CVResult:
    """Cross-validate a model trainer with shared fold splits."""
    trainer_kwargs = trainer_kwargs or {}
    y_array = np.asarray(y_train)
    oof_probabilities = np.zeros(len(y_array), dtype=float)
    test_probability_folds: list[np.ndarray] = []
    fold_models: list[Any] = []
    fold_metric_rows: list[dict[str, Any]] = []

    for fold_index, (train_idx, valid_idx) in enumerate(cv_splits):
        X_fold_train = X_train.iloc[train_idx]
        y_fold_train = y_array[train_idx]
        X_fold_valid = X_train.iloc[valid_idx]
        y_fold_valid = y_array[valid_idx]

        model = trainer(
            X_fold_train,
            y_fold_train,
            **trainer_kwargs,
        )
        valid_probabilities = predict_proba(model, X_fold_valid)
        valid_predictions = (valid_probabilities >= 0.5).astype(int)
        test_probabilities = predict_proba(model, X_test)

        oof_probabilities[valid_idx] = valid_probabilities
        test_probability_folds.append(test_probabilities)
        fold_models.append(model)

        fold_metrics = compute_fold_metrics(
            y_fold_valid,
            valid_predictions,
            valid_probabilities,
        )
        fold_metric_rows.append(
            {
                "model": model_name,
                "fold": fold_index,
                **fold_metrics,
            }
        )

    if np.isnan(oof_probabilities).any():
        raise ValueError(f"OOF probabilities contain NaN values for {model_name}")

    oof_predictions = (oof_probabilities >= 0.5).astype(int)
    averaged_test_probabilities = np.mean(test_probability_folds, axis=0)
    fold_metrics_df = pd.DataFrame(fold_metric_rows)

    return CVResult(
        model_name=model_name,
        oof_probabilities=oof_probabilities,
        oof_predictions=oof_predictions,
        test_probabilities=averaged_test_probabilities,
        fold_metrics=fold_metrics_df,
        fold_models=fold_models,
    )


def summarize_metrics(fold_metrics: pd.DataFrame) -> pd.DataFrame:
    """Summarize fold metrics with mean and standard deviation."""
    summary_rows: list[dict[str, Any]] = []
    for model_name, group in fold_metrics.groupby("model"):
        row: dict[str, Any] = {"model": model_name}
        for metric in METRIC_NAMES:
            row[f"mean_{metric}"] = float(group[metric].mean())
            row[f"std_{metric}"] = float(group[metric].std(ddof=0))
        summary_rows.append(row)
    return pd.DataFrame(summary_rows)


def print_metrics_summary(summary_df: pd.DataFrame) -> None:
    """Print mean ± std metrics for each model."""
    for _, row in summary_df.iterrows():
        print(f"{row['model']}:")
        for metric in METRIC_NAMES:
            mean_value = row[f"mean_{metric}"]
            std_value = row[f"std_{metric}"]
            print(f"  {metric}: {mean_value:.4f} +/- {std_value:.4f}")


def compare_models(summary_df: pd.DataFrame) -> pd.DataFrame:
    """Rank models by mean F1, precision, recall, and ROC AUC."""
    ranking_columns = [
        "mean_f1",
        "mean_precision",
        "mean_recall",
        "mean_roc_auc",
    ]
    missing = sorted(set(ranking_columns) - set(summary_df.columns))
    if missing:
        raise ValueError(f"Summary dataframe missing ranking columns: {missing}")
    return summary_df.sort_values(
        by=ranking_columns,
        ascending=[False, False, False, False],
    ).reset_index(drop=True)


def save_model(model: Any, path: Path | str) -> Path:
    """Persist a trained model with joblib."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_path)
    return output_path.resolve()


def load_model(path: Path | str) -> Any:
    """Load a persisted model from joblib."""
    model_path = Path(path)
    if not model_path.is_file():
        raise FileNotFoundError(f"Model artifact not found: {model_path}")
    return joblib.load(model_path)


def _encode_target(series: pd.Series) -> pd.Series:
    """Encode string target labels to binary integers."""
    encoded = series.map(TARGET_LABEL_TO_INT)
    if encoded.isna().any():
        invalid = series[encoded.isna()].unique().tolist()
        raise ValueError(f"Invalid target labels found: {invalid}")
    return encoded.astype(int)


def _extract_features(
    df: pd.DataFrame,
    target_column: str,
    id_column: str,
) -> pd.DataFrame:
    """Extract model feature matrix from an engineered dataframe."""
    exclude = {target_column, id_column}
    feature_columns = [column for column in df.columns if column not in exclude]
    if not feature_columns:
        raise ValueError("No feature columns available for model training")
    return df[feature_columns].copy()


def _load_feature_datasets(config: Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load engineered train and test feature datasets."""
    train_path = config.processed_data_dir / "train_features.parquet"
    test_path = config.processed_data_dir / "test_features.parquet"
    if not train_path.is_file():
        raise FileNotFoundError(f"Train features not found: {train_path}")
    if not test_path.is_file():
        raise FileNotFoundError(f"Test features not found: {test_path}")
    return pd.read_parquet(train_path), pd.read_parquet(test_path)


def _save_probability_frame(
    ids: pd.Series,
    probabilities: np.ndarray,
    path: Path,
    y_true: pd.Series | None = None,
    predictions: np.ndarray | None = None,
) -> Path:
    """Save probability outputs to CSV."""
    frame = pd.DataFrame(
        {
            "SEQN": ids.values,
            "proba_senior": probabilities,
        }
    )
    if y_true is not None:
        frame.insert(1, "y_true", y_true.values)
    if predictions is not None:
        frame["pred_label"] = predictions
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8")
    return path.resolve()


def _build_model_registry(
    random_state: int,
    scale_pos_weight: float,
) -> dict[str, dict[str, Any]]:
    """Return model trainers and trainer keyword arguments."""
    return {
        "catboost": {
            "trainer": train_catboost,
            "trainer_kwargs": {"random_state": random_state},
            "model_filename": "catboost_model.joblib",
        },
        "lightgbm": {
            "trainer": train_lightgbm,
            "trainer_kwargs": {"random_state": random_state},
            "model_filename": "lightgbm_model.joblib",
        },
        "xgboost": {
            "trainer": train_xgboost,
            "trainer_kwargs": {
                "random_state": random_state,
                "scale_pos_weight": scale_pos_weight,
            },
            "model_filename": "xgboost_model.joblib",
        },
        "extratrees": {
            "trainer": train_extratrees,
            "trainer_kwargs": {"random_state": random_state},
            "model_filename": "extratrees_model.joblib",
        },
    }


@timer
def main() -> None:
    """Train all models with cross-validation and persist artifacts."""
    config = Config()
    ensure_directories(config)
    setup_logging(config)
    set_random_seed(config.random_seed)

    print_section("Loading datasets...")
    train_df, test_df = _load_feature_datasets(config)
    X_train = _extract_features(
        train_df,
        config.target_column,
        config.id_column,
    )
    X_test = _extract_features(
        test_df,
        config.target_column,
        config.id_column,
    )
    y_train = _encode_target(train_df[config.target_column])
    train_ids = train_df[config.id_column]
    test_ids = test_df[config.id_column]

    class_counts = y_train.value_counts()
    scale_pos_weight = float(class_counts.get(0, 1) / max(class_counts.get(1, 1), 1))
    cv_splits = create_stratified_folds(
        y_train,
        n_folds=config.n_folds,
        random_state=config.random_seed,
        shuffle=True,
    )
    model_registry = _build_model_registry(
        config.random_seed,
        scale_pos_weight,
    )

    print_section("Training models...")
    cv_results: list[CVResult] = []
    for model_name, model_config in model_registry.items():
        logger.info("Cross-validating %s", model_name)
        result = cross_validate_model(
            trainer=model_config["trainer"],
            model_name=model_name,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            cv_splits=cv_splits,
            trainer_kwargs=model_config["trainer_kwargs"],
        )
        cv_results.append(result)

    all_fold_metrics = pd.concat(
        [result.fold_metrics for result in cv_results],
        ignore_index=True,
    )
    metrics_summary = summarize_metrics(all_fold_metrics)
    model_comparison = compare_models(metrics_summary)

    print_section("Saving outputs...")
    for result in cv_results:
        model_name = result.model_name
        _save_probability_frame(
            train_ids,
            result.oof_probabilities,
            config.output_dir / f"{model_name}_oof.csv",
            y_true=y_train,
            predictions=result.oof_predictions,
        )
        _save_probability_frame(
            test_ids,
            result.test_probabilities,
            config.output_dir / f"{model_name}_test.csv",
        )

        registry_entry = model_registry[model_name]
        full_model = registry_entry["trainer"](
            X_train,
            y_train,
            **registry_entry["trainer_kwargs"],
        )
        save_model(
            full_model,
            config.models_dir / registry_entry["model_filename"],
        )

    all_fold_metrics.to_csv(
        config.output_dir / "fold_metrics.csv",
        index=False,
        encoding="utf-8",
    )
    metrics_summary.to_csv(
        config.output_dir / "metrics_summary.csv",
        index=False,
        encoding="utf-8",
    )
    model_comparison.to_csv(
        config.output_dir / "model_comparison.csv",
        index=False,
        encoding="utf-8",
    )

    print_section("Model leaderboard:")
    for _, row in model_comparison.iterrows():
        print(
            f"{row['model']}: "
            f"F1={row['mean_f1']:.4f}, "
            f"Precision={row['mean_precision']:.4f}, "
            f"Recall={row['mean_recall']:.4f}, "
            f"ROC AUC={row['mean_roc_auc']:.4f}"
        )
    print_metrics_summary(metrics_summary)
    print_section("Completed successfully.")


if __name__ == "__main__":
    import sys

    train_models_module = ModuleType("train_models")
    train_models_module.__dict__.update(globals())
    sys.modules["train_models"] = train_models_module
    for _pickle_class in (CVResult,):
        _pickle_class.__module__ = "train_models"
    main()

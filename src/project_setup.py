"""Project foundation module for NHANES Age Prediction.

Provides configuration, logging, data loading, validation, cleaning,
and persistence utilities used across the project pipeline.
"""

from __future__ import annotations

import functools
import json
import logging
import os
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence, TypeVar

import numpy as np
import pandas as pd

F = TypeVar("F", bound=Callable[..., Any])

VALID_TARGET_LABELS: frozenset[str] = frozenset({"Adult", "Senior"})
TARGET_LABEL_TO_INT: dict[str, int] = {"Adult": 0, "Senior": 1}
LOW_VARIANCE_THRESHOLD: float = 1e-4


def get_project_root() -> Path:
    """Return the absolute path to the project root directory."""
    return Path(__file__).resolve().parent.parent


@dataclass
class Config:
    """Central configuration for paths, data files, and training settings."""

    project_root: Path = field(default_factory=get_project_root)
    random_seed: int = 42
    n_folds: int = 5
    target_column: str = "age_group"
    id_column: str = "SEQN"
    train_filename: str = "Train_dataset.csv"
    test_filename: str = "Test_dataset.csv"
    sample_submission_filename: str = "Sample_submission.csv"
    log_filename: str = "project.log"
    data_dir: Path = field(init=False)
    raw_data_dir: Path = field(init=False)
    processed_data_dir: Path = field(init=False)
    output_dir: Path = field(init=False)
    logs_dir: Path = field(init=False)
    models_dir: Path = field(init=False)
    figures_dir: Path = field(init=False)
    notebooks_dir: Path = field(init=False)
    submissions_dir: Path = field(init=False)
    train_file: Path = field(init=False)
    test_file: Path = field(init=False)
    sample_submission_file: Path = field(init=False)

    def __post_init__(self) -> None:
        """Resolve all directory and file paths from the project root."""
        self.project_root = Path(self.project_root).resolve()
        self.data_dir = self.project_root / "data"
        self.raw_data_dir = self.data_dir / "raw"
        self.processed_data_dir = self.data_dir / "processed"
        self.output_dir = self.project_root / "outputs"
        self.logs_dir = self.project_root / "logs"
        self.models_dir = self.project_root / "models"
        self.figures_dir = self.project_root / "figures"
        self.notebooks_dir = self.project_root / "notebooks"
        self.submissions_dir = self.project_root / "submissions"
        self.train_file = self.raw_data_dir / self.train_filename
        self.test_file = self.raw_data_dir / self.test_filename
        self.sample_submission_file = (
            self.raw_data_dir / self.sample_submission_filename
        )


def ensure_directories(config: Config) -> None:
    """Create all required project directories if they do not exist."""
    directories: tuple[Path, ...] = (
        config.data_dir,
        config.raw_data_dir,
        config.processed_data_dir,
        config.output_dir,
        config.logs_dir,
        config.models_dir,
        config.figures_dir,
        config.notebooks_dir,
        config.submissions_dir,
    )
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)


def setup_logging(config: Config, logger_name: str = "nhanes") -> logging.Logger:
    """Configure console and file logging.

    Args:
        config: Project configuration containing the logs directory.
        logger_name: Name assigned to the logger instance.

    Returns:
        Configured logger writing to console and ``logs/project.log``.
    """
    ensure_directories(config)
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    log_path = config.logs_dir / config.log_filename
    file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def set_random_seed(seed: int) -> None:
    """Set random seeds for reproducibility across supported libraries.

    Args:
        seed: Integer seed applied to ``random``, ``numpy``, and
            ``PYTHONHASHSEED``.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def resolve_data_path(config: Config, configured_path: Path) -> Path:
    """Resolve a data file path, checking raw data dir and project root.

    Args:
        config: Project configuration.
        configured_path: Primary expected location of the data file.

    Returns:
        Absolute path to an existing data file.

    Raises:
        FileNotFoundError: If the file cannot be located.
    """
    candidates = (
        configured_path,
        config.project_root / configured_path.name,
    )
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.is_file():
            return resolved
    searched = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"Data file not found. Searched: {searched}")


def _read_csv(path: Path) -> pd.DataFrame:
    """Read a CSV file with UTF-8 encoding.

    Args:
        path: Path to the CSV file.

    Returns:
        Loaded dataframe.

    Raises:
        FileNotFoundError: If the file does not exist.
        pd.errors.EmptyDataError: If the file contains no data.
        UnicodeDecodeError: If UTF-8 decoding fails.
    """
    if not path.is_file():
        raise FileNotFoundError(f"CSV file not found: {path}")
    return pd.read_csv(path, encoding="utf-8")


def load_train(config: Config) -> pd.DataFrame:
    """Load the training dataset.

    Args:
        config: Project configuration.

    Returns:
        Training dataframe.

    Raises:
        FileNotFoundError: If the training file is missing.
        ValueError: If required columns are absent or the dataset is empty.
    """
    path = resolve_data_path(config, config.train_file)
    df = _read_csv(path)
    if df.empty:
        raise ValueError(f"Training dataset is empty: {path}")
    required_columns = {config.id_column, config.target_column}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(
            f"Training dataset missing required columns: {sorted(missing)}"
        )
    return df


def load_test(config: Config) -> pd.DataFrame:
    """Load the test dataset.

    Args:
        config: Project configuration.

    Returns:
        Test dataframe.

    Raises:
        FileNotFoundError: If the test file is missing.
        ValueError: If required columns are absent or the dataset is empty.
    """
    path = resolve_data_path(config, config.test_file)
    df = _read_csv(path)
    if df.empty:
        raise ValueError(f"Test dataset is empty: {path}")
    if config.id_column not in df.columns:
        raise ValueError(
            f"Test dataset missing required column: {config.id_column}"
        )
    return df


def load_submission(config: Config) -> pd.DataFrame:
    """Load the sample submission file.

    Args:
        config: Project configuration.

    Returns:
        Sample submission dataframe.

    Raises:
        FileNotFoundError: If the submission file is missing.
        ValueError: If required columns are absent or the dataset is empty.
    """
    path = resolve_data_path(config, config.sample_submission_file)
    df = _read_csv(path)
    if df.empty:
        raise ValueError(f"Sample submission is empty: {path}")
    if config.target_column not in df.columns:
        raise ValueError(
            f"Sample submission missing required column: {config.target_column}"
        )
    return df


def load_all(config: Config) -> dict[str, pd.DataFrame]:
    """Load train, test, and sample submission datasets.

    Args:
        config: Project configuration.

    Returns:
        Dictionary with keys ``train``, ``test``, and ``submission``.
    """
    return {
        "train": load_train(config),
        "test": load_test(config),
        "submission": load_submission(config),
    }


def validate_schema(
    df: pd.DataFrame,
    required_columns: Sequence[str],
    dataset_name: str,
) -> dict[str, Any]:
    """Validate that a dataframe contains required columns.

    Args:
        df: Dataframe to validate.
        required_columns: Column names that must be present.
        dataset_name: Human-readable dataset identifier for reporting.

    Returns:
        Validation report dictionary.

    Raises:
        ValueError: If required columns are missing.
    """
    present_columns = list(df.columns)
    missing_columns = sorted(set(required_columns) - set(present_columns))
    extra_columns = sorted(set(present_columns) - set(required_columns))
    is_valid = not missing_columns
    report = {
        "dataset_name": dataset_name,
        "is_valid": is_valid,
        "row_count": len(df),
        "column_count": len(present_columns),
        "present_columns": present_columns,
        "missing_columns": missing_columns,
        "extra_columns": extra_columns,
    }
    if not is_valid:
        raise ValueError(
            f"{dataset_name} schema validation failed. "
            f"Missing columns: {missing_columns}"
        )
    return report


def validate_target(
    series: pd.Series,
    target_column: str,
    allow_numeric: bool = False,
) -> dict[str, Any]:
    """Validate target labels against expected class values.

    Args:
        series: Target series to validate.
        target_column: Name of the target column.
        allow_numeric: Whether integer labels ``0`` and ``1`` are valid.

    Returns:
        Validation report dictionary.

    Raises:
        ValueError: If invalid target labels are present.
    """
    non_null = series.dropna()
    unique_values = sorted(non_null.unique(), key=lambda value: str(value))
    if allow_numeric:
        allowed = {label for label in VALID_TARGET_LABELS}
        allowed.update({"0", "1", 0, 1})
    else:
        allowed = set(VALID_TARGET_LABELS)

    invalid_values = sorted(
        (
            value
            for value in unique_values
            if value not in allowed and str(value) not in allowed
        ),
        key=str,
    )
    missing_count = int(series.isna().sum())
    value_counts = non_null.value_counts(dropna=False).to_dict()

    report = {
        "target_column": target_column,
        "is_valid": not invalid_values,
        "unique_values": unique_values,
        "invalid_values": invalid_values,
        "missing_count": missing_count,
        "value_counts": value_counts,
    }
    if invalid_values:
        raise ValueError(
            f"Invalid target labels found in '{target_column}': {invalid_values}"
        )
    return report


def check_duplicate_rows(df: pd.DataFrame, dataset_name: str) -> dict[str, Any]:
    """Detect fully duplicated rows in a dataframe.

    Args:
        df: Dataframe to inspect.
        dataset_name: Human-readable dataset identifier.

    Returns:
        Duplicate row report dictionary.
    """
    duplicate_mask = df.duplicated(keep=False)
    duplicate_count = int(duplicate_mask.sum())
    duplicate_groups = int(df.duplicated().sum())
    return {
        "dataset_name": dataset_name,
        "duplicate_row_count": duplicate_count,
        "duplicate_group_count": duplicate_groups,
        "has_duplicates": duplicate_groups > 0,
    }


def check_duplicate_ids(
    df: pd.DataFrame,
    id_column: str,
    dataset_name: str,
) -> dict[str, Any]:
    """Detect duplicate identifier values.

    Args:
        df: Dataframe to inspect.
        id_column: Identifier column name.
        dataset_name: Human-readable dataset identifier.

    Returns:
        Duplicate identifier report dictionary.

    Raises:
        ValueError: If the identifier column is missing.
    """
    if id_column not in df.columns:
        raise ValueError(
            f"{dataset_name} missing identifier column '{id_column}'"
        )
    duplicated_ids = df[id_column][df[id_column].duplicated(keep=False)]
    duplicate_count = int(duplicated_ids.shape[0])
    duplicate_unique_count = int(duplicated_ids.nunique(dropna=False))
    return {
        "dataset_name": dataset_name,
        "id_column": id_column,
        "duplicate_id_row_count": duplicate_count,
        "duplicate_unique_id_count": duplicate_unique_count,
        "has_duplicate_ids": duplicate_unique_count > 0,
        "duplicated_ids": sorted(duplicated_ids.unique().tolist()),
    }


def missing_value_report(df: pd.DataFrame) -> pd.DataFrame:
    """Compute missing value counts and percentages by column.

    Args:
        df: Dataframe to analyze.

    Returns:
        Report dataframe sorted by descending missing count.
    """
    missing_count = df.isna().sum()
    missing_pct = df.isna().mean() * 100.0
    report = pd.DataFrame(
        {
            "column": df.columns,
            "dtype": df.dtypes.astype(str).values,
            "missing_count": missing_count.values,
            "missing_pct": missing_pct.values,
            "non_missing_count": (len(df) - missing_count).values,
        }
    )
    return report.sort_values(
        by=["missing_count", "column"],
        ascending=[False, True],
    ).reset_index(drop=True)


def numerical_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Generate descriptive statistics for numeric columns.

    Args:
        df: Dataframe to summarize.

    Returns:
        Summary dataframe for numeric columns.
    """
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.empty:
        return pd.DataFrame(
            columns=[
                "column",
                "count",
                "mean",
                "std",
                "min",
                "25%",
                "50%",
                "75%",
                "max",
                "missing_count",
                "missing_pct",
            ]
        )

    describe_df = numeric_df.describe(percentiles=[0.25, 0.5, 0.75]).T
    describe_df.index.name = "column"
    describe_df = describe_df.reset_index()
    describe_df["missing_count"] = numeric_df.isna().sum().values
    describe_df["missing_pct"] = numeric_df.isna().mean().values * 100.0
    return describe_df


def categorical_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Generate summary statistics for non-numeric columns.

    Args:
        df: Dataframe to summarize.

    Returns:
        Summary dataframe for categorical columns.
    """
    categorical_df = df.select_dtypes(exclude=[np.number])
    if categorical_df.empty:
        return pd.DataFrame(
            columns=[
                "column",
                "count",
                "unique",
                "top",
                "freq",
                "missing_count",
                "missing_pct",
            ]
        )

    summary_rows: list[dict[str, Any]] = []
    for column in categorical_df.columns:
        series = categorical_df[column]
        value_counts = series.value_counts(dropna=False)
        top_value = value_counts.index[0] if not value_counts.empty else np.nan
        top_freq = int(value_counts.iloc[0]) if not value_counts.empty else 0
        summary_rows.append(
            {
                "column": column,
                "count": int(series.count()),
                "unique": int(series.nunique(dropna=False)),
                "top": top_value,
                "freq": top_freq,
                "missing_count": int(series.isna().sum()),
                "missing_pct": float(series.isna().mean() * 100.0),
            }
        )
    return pd.DataFrame(summary_rows)


def unique_value_report(df: pd.DataFrame) -> pd.DataFrame:
    """Report unique value counts for every column.

    Args:
        df: Dataframe to analyze.

    Returns:
        Report dataframe with unique counts per column.
    """
    rows = [
        {
            "column": column,
            "unique_count": int(df[column].nunique(dropna=False)),
            "non_null_count": int(df[column].notna().sum()),
        }
        for column in df.columns
    ]
    return pd.DataFrame(rows).sort_values(
        by=["unique_count", "column"],
        ascending=[True, True],
    ).reset_index(drop=True)


def memory_usage(df: pd.DataFrame) -> dict[str, Any]:
    """Compute dataframe memory usage statistics.

    Args:
        df: Dataframe to inspect.

    Returns:
        Memory usage report dictionary.
    """
    usage_bytes = int(df.memory_usage(deep=True).sum())
    column_usage = df.memory_usage(deep=True).sort_values(ascending=False)
    column_usage_dict = {
        str(column): int(value) for column, value in column_usage.items()
    }
    return {
        "total_bytes": usage_bytes,
        "total_human": format_bytes(usage_bytes),
        "row_count": len(df),
        "column_count": len(df.columns),
        "column_usage_bytes": column_usage_dict,
    }


def constant_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Identify columns with a single unique non-null value.

    Args:
        df: Dataframe to inspect.

    Returns:
        Dataframe listing constant columns and their values.
    """
    rows: list[dict[str, Any]] = []
    for column in df.columns:
        unique_count = df[column].nunique(dropna=False)
        if unique_count <= 1:
            unique_values = df[column].dropna().unique().tolist()
            rows.append(
                {
                    "column": column,
                    "unique_count": int(unique_count),
                    "constant_value": unique_values[0] if unique_values else np.nan,
                }
            )
    return pd.DataFrame(rows)


def low_variance_columns(
    df: pd.DataFrame,
    threshold: float = LOW_VARIANCE_THRESHOLD,
) -> pd.DataFrame:
    """Identify numeric columns with variance below a threshold.

    Args:
        df: Dataframe to inspect.
        threshold: Maximum variance considered low.

    Returns:
        Dataframe listing low-variance numeric columns.
    """
    numeric_df = df.select_dtypes(include=[np.number])
    rows: list[dict[str, Any]] = []
    for column in numeric_df.columns:
        series = numeric_df[column].dropna()
        if series.empty:
            variance = 0.0
        else:
            variance = float(series.var(ddof=0))
        if variance <= threshold:
            rows.append(
                {
                    "column": column,
                    "variance": variance,
                    "threshold": threshold,
                }
            )
    return pd.DataFrame(rows)


def compare_train_test_columns(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_column: str,
    id_column: str,
) -> dict[str, Any]:
    """Compare feature columns between train and test datasets.

    Args:
        train_df: Training dataframe.
        test_df: Test dataframe.
        target_column: Target column excluded from feature comparison.
        id_column: Identifier column excluded from feature comparison.

    Returns:
        Comparison report dictionary.
    """
    excluded = {target_column, id_column}
    train_features = sorted(set(train_df.columns) - excluded)
    test_features = sorted(set(test_df.columns) - excluded)
    train_only = sorted(set(train_features) - set(test_features))
    test_only = sorted(set(test_features) - set(train_features))
    shared = sorted(set(train_features) & set(test_features))
    return {
        "train_feature_count": len(train_features),
        "test_feature_count": len(test_features),
        "shared_feature_count": len(shared),
        "shared_features": shared,
        "train_only_features": train_only,
        "test_only_features": test_only,
        "columns_match": not train_only and not test_only,
    }


def dataset_overview(
    df: pd.DataFrame,
    dataset_name: str,
) -> dict[str, Any]:
    """Generate a structured overview for a dataframe.

    Args:
        df: Dataframe to summarize.
        dataset_name: Human-readable dataset identifier.

    Returns:
        Overview dictionary with shape, dtypes, and duplicate information.
    """
    duplicate_rows = check_duplicate_rows(df, dataset_name)
    return {
        "dataset_name": dataset_name,
        "row_count": len(df),
        "column_count": len(df.columns),
        "columns": list(df.columns),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "numeric_column_count": int(df.select_dtypes(include=[np.number]).shape[1]),
        "categorical_column_count": int(
            df.select_dtypes(exclude=[np.number]).shape[1]
        ),
        "duplicate_row_count": duplicate_rows["duplicate_row_count"],
        "duplicate_group_count": duplicate_rows["duplicate_group_count"],
    }


def remove_missing_target(
    df: pd.DataFrame,
    target_column: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Remove rows with missing target values.

    Args:
        df: Input dataframe.
        target_column: Target column name.

    Returns:
        Cleaned dataframe and a report describing removed rows.

    Raises:
        ValueError: If the target column is missing.
    """
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataframe")
    missing_mask = df[target_column].isna()
    removed_count = int(missing_mask.sum())
    cleaned_df = df.loc[~missing_mask].copy()
    report = {
        "target_column": target_column,
        "initial_row_count": len(df),
        "removed_row_count": removed_count,
        "final_row_count": len(cleaned_df),
    }
    return cleaned_df, report


def split_features_target(
    df: pd.DataFrame,
    target_column: str,
) -> tuple[pd.DataFrame, pd.Series]:
    """Split a dataframe into features and target.

    Args:
        df: Input dataframe containing the target column.
        target_column: Name of the target column.

    Returns:
        Feature dataframe and target series.

    Raises:
        ValueError: If the target column is missing.
    """
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataframe")
    features = df.drop(columns=[target_column]).copy()
    target = df[target_column].copy()
    return features, target


def extract_id_column(
    df: pd.DataFrame,
    id_column: str,
) -> tuple[pd.Series, pd.DataFrame]:
    """Extract the identifier column from a dataframe.

    Args:
        df: Input dataframe.
        id_column: Identifier column name.

    Returns:
        Identifier series and dataframe without the identifier column.

    Raises:
        ValueError: If the identifier column is missing.
    """
    if id_column not in df.columns:
        raise ValueError(f"Identifier column '{id_column}' not found in dataframe")
    identifiers = df[id_column].copy()
    features = df.drop(columns=[id_column]).copy()
    return identifiers, features


def drop_id_column(df: pd.DataFrame, id_column: str) -> pd.DataFrame:
    """Return a copy of the dataframe without the identifier column.

    Args:
        df: Input dataframe.
        id_column: Identifier column name.

    Returns:
        Dataframe without the identifier column.

    Raises:
        ValueError: If the identifier column is missing.
    """
    if id_column not in df.columns:
        raise ValueError(f"Identifier column '{id_column}' not found in dataframe")
    return df.drop(columns=[id_column]).copy()


def restore_id_column(
    df: pd.DataFrame,
    identifiers: pd.Series,
    id_column: str,
) -> pd.DataFrame:
    """Attach an identifier column to a dataframe.

    Args:
        df: Feature dataframe.
        identifiers: Identifier values aligned with ``df`` rows.
        id_column: Identifier column name.

    Returns:
        Dataframe with the identifier column prepended.

    Raises:
        ValueError: If row counts do not align.
    """
    if len(df) != len(identifiers):
        raise ValueError(
            "Identifier length does not match dataframe row count: "
            f"{len(identifiers)} != {len(df)}"
        )
    restored = df.copy()
    restored.insert(0, id_column, identifiers.values)
    return restored


def save_dataframe(
    df: pd.DataFrame,
    path: Path | str,
    index: bool = False,
    **kwargs: Any,
) -> Path:
    """Save a dataframe to CSV, creating parent directories if needed.

    Args:
        df: Dataframe to persist.
        path: Destination file path.
        index: Whether to include the dataframe index in the output.
        **kwargs: Additional arguments forwarded to ``DataFrame.to_csv``.

    Returns:
        Absolute path to the saved file.
    """
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=index, encoding="utf-8", **kwargs)
    return output_path.resolve()


def save_text(content: str, path: Path | str) -> Path:
    """Save plain text content to a file.

    Args:
        content: Text content to write.
        path: Destination file path.

    Returns:
        Absolute path to the saved file.
    """
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content, encoding="utf-8")
    return output_path.resolve()


def save_json(data: Any, path: Path | str, indent: int = 2) -> Path:
    """Save JSON-serializable data to a file.

    Args:
        data: JSON-serializable object.
        path: Destination file path.
        indent: JSON indentation level.

    Returns:
        Absolute path to the saved file.
    """
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=indent, default=str)
    return output_path.resolve()


def timer(func: F) -> F:
    """Decorator that logs execution time for a callable."""

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        logging.getLogger("nhanes").info(
            "%s completed in %.4f seconds", func.__name__, elapsed
        )
        return result

    return wrapper  # type: ignore[return-value]


def print_section(title: str) -> None:
    """Print a concise section header to stdout.

    Args:
        title: Section title text.
    """
    print(title)


def format_bytes(num_bytes: int) -> str:
    """Convert a byte count to a human-readable string.

    Args:
        num_bytes: Number of bytes.

    Returns:
        Formatted size string.
    """
    if num_bytes < 0:
        raise ValueError("Byte count must be non-negative")
    units = ("B", "KB", "MB", "GB", "TB")
    size = float(num_bytes)
    for unit in units:
        if size < 1024.0 or unit == units[-1]:
            return f"{size:.2f} {unit}"
        size /= 1024.0
    return f"{size:.2f} TB"


def safe_divide(
    numerator: float,
    denominator: float,
    default: float = 0.0,
) -> float:
    """Safely divide two numbers, returning a default when denominator is zero.

    Args:
        numerator: Dividend value.
        denominator: Divisor value.
        default: Value returned when the denominator is zero.

    Returns:
        Division result or the default value.
    """
    if denominator == 0:
        return default
    return numerator / denominator


def _format_overview_text(overviews: Iterable[dict[str, Any]]) -> str:
    """Format dataset overview dictionaries as plain text."""
    lines: list[str] = []
    for overview in overviews:
        lines.append(f"Dataset: {overview['dataset_name']}")
        lines.append(f"Rows: {overview['row_count']}")
        lines.append(f"Columns: {overview['column_count']}")
        lines.append(
            "Numeric columns: "
            f"{overview['numeric_column_count']} | "
            f"Categorical columns: {overview['categorical_column_count']}"
        )
        lines.append(
            "Duplicate rows: "
            f"{overview['duplicate_row_count']} "
            f"(groups: {overview['duplicate_group_count']})"
        )
        lines.append("Column names:")
        lines.extend(f"  - {column}" for column in overview["columns"])
        lines.append("Dtypes:")
        for column, dtype in overview["dtypes"].items():
            lines.append(f"  - {column}: {dtype}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _format_memory_text(memory_reports: dict[str, dict[str, Any]]) -> str:
    """Format memory usage reports as plain text."""
    lines: list[str] = []
    for dataset_name, report in memory_reports.items():
        lines.append(f"Dataset: {dataset_name}")
        lines.append(f"Total memory: {report['total_human']} ({report['total_bytes']} bytes)")
        lines.append(f"Rows: {report['row_count']} | Columns: {report['column_count']}")
        lines.append("Column memory usage:")
        for column, value in report["column_usage_bytes"].items():
            lines.append(f"  - {column}: {format_bytes(value)}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _format_comparison_text(comparison: dict[str, Any]) -> str:
    """Format train/test column comparison as plain text."""
    lines = [
        f"Train feature count: {comparison['train_feature_count']}",
        f"Test feature count: {comparison['test_feature_count']}",
        f"Shared feature count: {comparison['shared_feature_count']}",
        f"Columns match: {comparison['columns_match']}",
        "",
        "Shared features:",
    ]
    lines.extend(f"  - {column}" for column in comparison["shared_features"])
    lines.extend(["", "Train-only features:"])
    if comparison["train_only_features"]:
        lines.extend(
            f"  - {column}" for column in comparison["train_only_features"]
        )
    else:
        lines.append("  - None")
    lines.extend(["", "Test-only features:"])
    if comparison["test_only_features"]:
        lines.extend(
            f"  - {column}" for column in comparison["test_only_features"]
        )
    else:
        lines.append("  - None")
    return "\n".join(lines) + "\n"


def _class_distribution_df(
    series: pd.Series,
    target_column: str,
) -> pd.DataFrame:
    """Build a class distribution dataframe from a target series."""
    counts = series.value_counts(dropna=False)
    total = max(len(series), 1)
    rows = [
        {
            "target_column": target_column,
            "class_label": label,
            "count": int(count),
            "percentage": safe_divide(float(count), float(total)) * 100.0,
            "encoded_label": TARGET_LABEL_TO_INT.get(str(label), np.nan),
        }
        for label, count in counts.items()
    ]
    return pd.DataFrame(rows)


@timer
def main() -> None:
    """Run the Phase 1 dataset audit and persist validation artifacts."""
    config = Config()
    ensure_directories(config)
    logger = setup_logging(config)
    set_random_seed(config.random_seed)

    print_section("Loading datasets...")
    datasets = load_all(config)
    train_df = datasets["train"]
    test_df = datasets["test"]

    print_section("Checking schema...")
    validate_schema(
        train_df,
        required_columns=[config.id_column, config.target_column],
        dataset_name="train",
    )
    validate_schema(
        test_df,
        required_columns=[config.id_column],
        dataset_name="test",
    )
    validate_target(train_df[config.target_column], config.target_column)

    print_section("Generating summaries...")
    duplicate_rows_train = check_duplicate_rows(train_df, "train")
    duplicate_rows_test = check_duplicate_rows(test_df, "test")
    duplicate_ids_train = check_duplicate_ids(
        train_df,
        config.id_column,
        "train",
    )
    duplicate_ids_test = check_duplicate_ids(
        test_df,
        config.id_column,
        "test",
    )

    missing_values = missing_value_report(train_df)
    unique_values = unique_value_report(train_df)
    numeric_summary = numerical_summary(train_df)
    categorical_summary_df = categorical_summary(train_df)
    memory_report_train = memory_usage(train_df)
    memory_report_test = memory_usage(test_df)
    constant_cols = constant_columns(train_df)
    low_variance_cols = low_variance_columns(train_df)
    overview_train = dataset_overview(train_df, "train")
    overview_test = dataset_overview(test_df, "test")

    cleaned_train_df, target_cleanup_report = remove_missing_target(
        train_df,
        config.target_column,
    )
    comparison = compare_train_test_columns(
        cleaned_train_df,
        test_df,
        config.target_column,
        config.id_column,
    )
    class_distribution = _class_distribution_df(
        cleaned_train_df[config.target_column],
        config.target_column,
    )

    logger.info("Unique value report computed for %s columns", len(unique_values))
    logger.info("Removed %s rows with missing target", target_cleanup_report["removed_row_count"])
    logger.info("Train duplicate rows: %s", duplicate_rows_train["duplicate_group_count"])
    logger.info("Test duplicate rows: %s", duplicate_rows_test["duplicate_group_count"])
    logger.info("Train duplicate IDs: %s", duplicate_ids_train["has_duplicate_ids"])
    logger.info("Test duplicate IDs: %s", duplicate_ids_test["has_duplicate_ids"])

    print_section("Saving outputs...")
    output_dir = config.output_dir
    save_dataframe(missing_values, output_dir / "missing_values.csv")
    save_dataframe(numeric_summary, output_dir / "numerical_summary.csv")
    save_dataframe(categorical_summary_df, output_dir / "categorical_summary.csv")
    save_dataframe(class_distribution, output_dir / "class_distribution.csv")
    save_dataframe(constant_cols, output_dir / "constant_columns.csv")
    save_dataframe(low_variance_cols, output_dir / "low_variance_columns.csv")
    save_text(
        _format_overview_text([overview_train, overview_test]),
        output_dir / "dataset_overview.txt",
    )
    save_text(
        _format_memory_text(
            {
                "train": memory_report_train,
                "test": memory_report_test,
            }
        ),
        output_dir / "memory_usage.txt",
    )
    save_text(
        _format_comparison_text(comparison),
        output_dir / "train_test_comparison.txt",
    )

    print_section("Completed successfully.")


if __name__ == "__main__":
    main()

"""Advanced feature engineering for NHANES Age Prediction."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from types import ModuleType
from typing import Any, Sequence

import joblib
import numpy as np
import pandas as pd

from preprocessing import PreprocessingPipeline
from project_setup import (
    Config,
    ensure_directories,
    load_test,
    load_train,
    print_section,
    remove_missing_target,
    safe_divide,
    set_random_seed,
    setup_logging,
    timer,
)

logger = logging.getLogger("nhanes")

BMI_COLUMN = "BMXBMI"
GLUCOSE_COLUMN = "LBXGLU"
GLUCOSE_TOLERANCE_COLUMN = "LBXGLT"
INSULIN_COLUMN = "LBXIN"

INTERACTION_SPECS: tuple[tuple[str, str, str], ...] = (
    (BMI_COLUMN, GLUCOSE_COLUMN, "BMXBMI_x_LBXGLU"),
    (BMI_COLUMN, INSULIN_COLUMN, "BMXBMI_x_LBXIN"),
    (BMI_COLUMN, GLUCOSE_TOLERANCE_COLUMN, "BMXBMI_x_LBXGLT"),
    (GLUCOSE_COLUMN, INSULIN_COLUMN, "LBXGLU_x_LBXIN"),
    (GLUCOSE_COLUMN, GLUCOSE_TOLERANCE_COLUMN, "LBXGLU_x_LBXGLT"),
    (INSULIN_COLUMN, GLUCOSE_TOLERANCE_COLUMN, "LBXIN_x_LBXGLT"),
)

DIFFERENCE_SPECS: tuple[tuple[str, str, str], ...] = (
    (GLUCOSE_TOLERANCE_COLUMN, GLUCOSE_COLUMN, "LBXGLT_minus_LBXGLU"),
)

RATIO_SPECS: tuple[tuple[str, str, str], ...] = (
    (GLUCOSE_TOLERANCE_COLUMN, GLUCOSE_COLUMN, "LBXGLT_div_LBXGLU"),
    (INSULIN_COLUMN, GLUCOSE_COLUMN, "LBXIN_div_LBXGLU"),
    (BMI_COLUMN, GLUCOSE_COLUMN, "BMXBMI_div_LBXGLU"),
)


@dataclass
class FeatureEngineeringConfig:
    """Configuration options for ``FeatureEngineer``."""

    target_column: str = "age_group"
    id_column: str = "SEQN"
    enable_interactions: bool = True
    enable_differences: bool = True
    enable_ratios: bool = True
    enable_log: bool = True
    enable_polynomial: bool = False
    enable_quantile_bins: bool = True
    enable_missing_indicators: bool = True
    enable_statistical: bool = True
    quantile_bins: int = 4
    log_skew_threshold: float = 1.0
    binary_unique_threshold: int = 2


@dataclass
class QuantileBinEdges:
    """Quantile bin edges fitted on a continuous feature."""

    column: str
    bin_edges: np.ndarray


class FeatureEngineer:
    """Fit-transform feature engineering pipeline with persistence support."""

    def __init__(
        self,
        config: FeatureEngineeringConfig | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the feature engineer.

        Args:
            config: Optional feature engineering configuration.
            **kwargs: Keyword arguments forwarded to ``FeatureEngineeringConfig``.
        """
        self.config = (
            config if config is not None else FeatureEngineeringConfig(**kwargs)
        )
        self.original_features_: list[str] = []
        self.generated_features_: list[str] = []
        self.numeric_features_: list[str] = []
        self.continuous_features_: list[str] = []
        self.log_features_: list[str] = []
        self.missing_indicator_columns_: list[str] = []
        self.quantile_bin_edges_: dict[str, np.ndarray] = {}
        self.is_fitted_: bool = False

    @property
    def metadata_columns(self) -> set[str]:
        """Return non-feature metadata column names."""
        return {self.config.id_column, self.config.target_column}

    def get_feature_names(self) -> list[str]:
        """Return all model feature names including generated features."""
        self._ensure_fitted()
        return self.original_features_ + self.generated_features_

    def get_generated_features(self) -> list[str]:
        """Return names of generated feature columns only."""
        self._ensure_fitted()
        return list(self.generated_features_)

    def _ensure_fitted(self) -> None:
        """Raise when the engineer has not been fitted."""
        if not self.is_fitted_:
            raise RuntimeError("FeatureEngineer must be fitted before use")

    @staticmethod
    def _available_columns(columns: Sequence[str], df: pd.DataFrame) -> list[str]:
        """Return column names present in the dataframe."""
        return [column for column in columns if column in df.columns]

    def _identify_numeric_features(self, df: pd.DataFrame) -> list[str]:
        """Detect numeric feature columns excluding metadata fields."""
        numeric_columns: list[str] = []
        for column in df.columns:
            if column in self.metadata_columns:
                continue
            if pd.api.types.is_numeric_dtype(df[column]):
                numeric_columns.append(column)
        return numeric_columns

    def _identify_continuous_features(
        self,
        df: pd.DataFrame,
        numeric_columns: Sequence[str],
    ) -> list[str]:
        """Detect continuous numeric columns excluding binary fields."""
        continuous_columns: list[str] = []
        for column in numeric_columns:
            unique_count = int(df[column].nunique(dropna=True))
            if unique_count > self.config.binary_unique_threshold:
                continuous_columns.append(column)
        return continuous_columns

    def _align_raw_dataframe(
        self,
        processed_df: pd.DataFrame,
        raw_df: pd.DataFrame | None,
    ) -> pd.DataFrame | None:
        """Align raw dataframe rows to processed dataframe order."""
        if raw_df is None:
            return None
        if len(processed_df) != len(raw_df):
            raise ValueError(
                "Processed and raw dataframes must have identical row counts "
                f"for alignment: {len(processed_df)} != {len(raw_df)}"
            )
        aligned = raw_df.reset_index(drop=True).copy()
        if self.config.id_column in processed_df.columns and (
            self.config.id_column in aligned.columns
        ):
            processed_ids = processed_df[self.config.id_column]
            raw_ids = aligned[self.config.id_column]
            mismatched = processed_ids.ne(raw_ids) & processed_ids.notna() & raw_ids.notna()
            if mismatched.any():
                mismatch_count = int(mismatched.sum())
                logger.warning(
                    "Identifier mismatch on %s aligned rows; using positional alignment",
                    mismatch_count,
                )
        return aligned

    def _identify_preprocessing_missing_columns(
        self,
        raw_df: pd.DataFrame,
        feature_columns: Sequence[str],
    ) -> list[str]:
        """Detect feature columns with missing values before preprocessing."""
        missing_columns: list[str] = []
        for column in feature_columns:
            if column not in raw_df.columns:
                continue
            if raw_df[column].isna().any():
                missing_columns.append(column)
        return missing_columns

    def _identify_log_columns(self, df: pd.DataFrame) -> list[str]:
        """Select positive-skewed continuous columns for log1p transformation."""
        log_columns: list[str] = []
        for column in self.continuous_features_:
            series = df[column].dropna()
            if series.empty:
                continue
            if float(series.min()) <= -1.0:
                continue
            if float(series.skew()) >= self.config.log_skew_threshold:
                log_columns.append(column)
        return log_columns

    def _fit_quantile_bins(self, df: pd.DataFrame) -> None:
        """Fit quantile bin edges for continuous columns."""
        self.quantile_bin_edges_.clear()
        for column in self.continuous_features_:
            series = df[column].dropna()
            if series.nunique(dropna=True) < self.config.quantile_bins:
                continue
            try:
                _, bin_edges = pd.qcut(
                    series,
                    q=self.config.quantile_bins,
                    retbins=True,
                    duplicates="drop",
                )
            except ValueError as exc:
                raise ValueError(
                    f"Unable to fit quantile bins for column '{column}'"
                ) from exc
            self.quantile_bin_edges_[column] = np.asarray(bin_edges)

    def _generate_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate configured metabolic interaction features."""
        generated = pd.DataFrame(index=df.index)
        if not self.config.enable_interactions:
            return generated
        for left, right, name in INTERACTION_SPECS:
            if left not in df.columns or right not in df.columns:
                continue
            generated[name] = df[left].astype(float) * df[right].astype(float)
        return generated

    def _generate_difference_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate configured difference features."""
        generated = pd.DataFrame(index=df.index)
        if not self.config.enable_differences:
            return generated
        for minuend, subtrahend, name in DIFFERENCE_SPECS:
            if minuend not in df.columns or subtrahend not in df.columns:
                continue
            generated[name] = df[minuend].astype(float) - df[subtrahend].astype(float)
        return generated

    def _generate_ratio_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate configured ratio features using safe division."""
        generated = pd.DataFrame(index=df.index)
        if not self.config.enable_ratios:
            return generated
        for numerator_col, denominator_col, name in RATIO_SPECS:
            if numerator_col not in df.columns or denominator_col not in df.columns:
                continue
            numerator = df[numerator_col].astype(float)
            denominator = df[denominator_col].astype(float)
            generated[name] = [
                safe_divide(float(num), float(den), default=0.0)
                for num, den in zip(numerator, denominator, strict=True)
            ]
        return generated

    def _generate_log_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate log1p features for skewed continuous columns."""
        generated = pd.DataFrame(index=df.index)
        if not self.config.enable_log:
            return generated
        for column in self.log_features_:
            if column not in df.columns:
                continue
            values = df[column].astype(float).clip(lower=-0.999999)
            generated[f"{column}_log1p"] = np.log1p(values)
        return generated

    def _generate_polynomial_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate optional degree-2 polynomial features."""
        generated = pd.DataFrame(index=df.index)
        if not self.config.enable_polynomial:
            return generated
        for column in self.continuous_features_:
            if column not in df.columns:
                continue
            generated[f"{column}_sq"] = df[column].astype(float) ** 2
        for left, right in combinations(self.continuous_features_, 2):
            if left not in df.columns or right not in df.columns:
                continue
            generated[f"{left}_x_{right}"] = (
                df[left].astype(float) * df[right].astype(float)
            )
        return generated

    def _generate_quantile_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate quantile bin features using fitted bin edges."""
        generated = pd.DataFrame(index=df.index)
        if not self.config.enable_quantile_bins:
            return generated
        for column, bin_edges in self.quantile_bin_edges_.items():
            if column not in df.columns:
                continue
            binned = pd.cut(
                df[column].astype(float),
                bins=bin_edges,
                include_lowest=True,
                labels=False,
            )
            generated[f"{column}_quantile_bin"] = binned.astype(float)
        return generated

    def _generate_missing_indicators(
        self,
        raw_df: pd.DataFrame | None,
        index: pd.Index,
    ) -> pd.DataFrame:
        """Generate binary indicators for pre-preprocessing missing values."""
        generated = pd.DataFrame(index=index)
        if not self.config.enable_missing_indicators or raw_df is None:
            return generated
        for column in self.missing_indicator_columns_:
            if column not in raw_df.columns:
                generated[f"{column}_missing"] = 0.0
            else:
                generated[f"{column}_missing"] = raw_df[column].isna().astype(float)
        return generated

    def _generate_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate row-wise statistical features over continuous columns."""
        generated = pd.DataFrame(index=df.index)
        if not self.config.enable_statistical:
            return generated
        continuous = self._available_columns(self.continuous_features_, df)
        if not continuous:
            return generated
        continuous_df = df[continuous].astype(float)
        generated["row_mean_continuous"] = continuous_df.mean(axis=1)
        generated["row_median_continuous"] = continuous_df.median(axis=1)
        generated["row_std_continuous"] = continuous_df.std(axis=1, ddof=0)
        generated["row_max_continuous"] = continuous_df.max(axis=1)
        generated["row_min_continuous"] = continuous_df.min(axis=1)
        generated["row_range_continuous"] = (
            generated["row_max_continuous"] - generated["row_min_continuous"]
        )
        return generated

    def _assemble_generated_features(
        self,
        feature_df: pd.DataFrame,
        raw_df: pd.DataFrame | None,
    ) -> pd.DataFrame:
        """Create all configured generated feature blocks."""
        blocks = [
            self._generate_interaction_features(feature_df),
            self._generate_difference_features(feature_df),
            self._generate_ratio_features(feature_df),
            self._generate_log_features(feature_df),
            self._generate_polynomial_features(feature_df),
            self._generate_quantile_features(feature_df),
            self._generate_missing_indicators(raw_df, feature_df.index),
            self._generate_statistical_features(feature_df),
        ]
        if not blocks:
            return pd.DataFrame(index=feature_df.index)
        generated = pd.concat(blocks, axis=1)
        return generated.loc[:, ~generated.columns.duplicated()]

    def _extract_feature_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return numeric feature columns from an input dataframe."""
        PreprocessingPipeline.check_duplicate_columns(df, dataset_name="input")
        numeric_columns = self._identify_numeric_features(df)
        if not numeric_columns:
            raise ValueError("No numeric feature columns found for engineering")
        return df[numeric_columns].copy()

    def _validate_generated_features(
        self,
        generated: pd.DataFrame,
        *,
        check_constants: bool,
    ) -> None:
        """Validate generated feature quality."""
        if generated.empty:
            return
        if generated.columns.duplicated().any():
            duplicated = generated.columns[
                generated.columns.duplicated()
            ].tolist()
            raise ValueError(f"Duplicate generated feature columns: {duplicated}")

        numeric_generated = generated.select_dtypes(include=[np.number])
        if numeric_generated.isna().any().any():
            nan_columns = numeric_generated.columns[
                numeric_generated.isna().any()
            ].tolist()
            raise ValueError(
                f"Generated features contain NaN values: {nan_columns}"
            )

        values = numeric_generated.to_numpy(dtype=float, copy=False)
        if np.isinf(values).any():
            inf_columns = numeric_generated.columns[
                np.isinf(numeric_generated.to_numpy()).any(axis=0)
            ].tolist()
            raise ValueError(
                f"Generated features contain infinite values: {inf_columns}"
            )

        if check_constants:
            constant_columns = [
                column
                for column in generated.columns
                if generated[column].nunique(dropna=False) <= 1
            ]
            if constant_columns:
                raise ValueError(
                    f"Generated constant feature columns: {constant_columns}"
                )

    def _compose_output(
        self,
        source_df: pd.DataFrame,
        generated: pd.DataFrame,
    ) -> pd.DataFrame:
        """Combine metadata, original features, and generated features."""
        output_columns: list[str] = []
        result = pd.DataFrame(index=source_df.index)

        if self.config.id_column in source_df.columns:
            result[self.config.id_column] = source_df[self.config.id_column]
            output_columns.append(self.config.id_column)

        for column in self.original_features_:
            result[column] = source_df[column]
            output_columns.append(column)

        for column in generated.columns:
            result[column] = generated[column]
            output_columns.append(column)

        if self.config.target_column in source_df.columns:
            result[self.config.target_column] = source_df[self.config.target_column]
            output_columns.append(self.config.target_column)

        PreprocessingPipeline.check_duplicate_columns(result, dataset_name="engineered")
        return result[output_columns]

    def fit(
        self,
        df: pd.DataFrame,
        raw_df: pd.DataFrame | None = None,
    ) -> FeatureEngineer:
        """Fit feature engineering state on training data.

        Args:
            df: Processed training dataframe.
            raw_df: Optional raw training dataframe for missing indicators.

        Returns:
            Fitted feature engineer instance.
        """
        feature_df = self._extract_feature_frame(df)
        aligned_raw = self._align_raw_dataframe(df, raw_df)

        self.original_features_ = list(feature_df.columns)
        self.numeric_features_ = list(feature_df.columns)
        self.continuous_features_ = self._identify_continuous_features(
            feature_df,
            self.numeric_features_,
        )
        self.log_features_ = self._identify_log_columns(feature_df)
        self.missing_indicator_columns_ = self._identify_preprocessing_missing_columns(
            aligned_raw if aligned_raw is not None else feature_df,
            self.original_features_,
        )
        self._fit_quantile_bins(feature_df)

        generated = self._assemble_generated_features(feature_df, aligned_raw)
        self._validate_generated_features(generated, check_constants=True)
        self.generated_features_ = list(generated.columns)
        self.is_fitted_ = True

        logger.info(
            "Fitted feature engineer on %s rows with %s generated features",
            len(df),
            len(self.generated_features_),
        )
        return self

    def transform(
        self,
        df: pd.DataFrame,
        raw_df: pd.DataFrame | None = None,
        dataset_name: str = "dataset",
    ) -> pd.DataFrame:
        """Transform a dataframe by applying fitted feature engineering.

        Args:
            df: Processed dataframe to transform.
            raw_df: Optional raw dataframe for missing indicators.
            dataset_name: Human-readable dataset identifier.

        Returns:
            Dataframe with original and generated features.
        """
        self._ensure_fitted()
        feature_df = self._extract_feature_frame(df)
        missing_columns = sorted(
            set(self.original_features_) - set(feature_df.columns)
        )
        if missing_columns:
            raise ValueError(
                f"{dataset_name} is missing expected feature columns: "
                f"{missing_columns}"
            )
        if list(feature_df.columns) != self.original_features_:
            feature_df = feature_df[self.original_features_]

        aligned_raw = self._align_raw_dataframe(df, raw_df)
        generated = self._assemble_generated_features(feature_df, aligned_raw)

        missing_generated = sorted(
            set(self.generated_features_) - set(generated.columns)
        )
        if missing_generated:
            raise ValueError(
                f"{dataset_name} failed to generate expected features: "
                f"{missing_generated}"
            )
        generated = generated[self.generated_features_]
        self._validate_generated_features(generated, check_constants=False)

        return self._compose_output(df, generated)

    def fit_transform(
        self,
        df: pd.DataFrame,
        raw_df: pd.DataFrame | None = None,
        dataset_name: str = "train",
    ) -> pd.DataFrame:
        """Fit feature engineering and transform the input dataframe."""
        self.fit(df, raw_df=raw_df)
        return self.transform(df, raw_df=raw_df, dataset_name=dataset_name)

    def save(self, path: Path | str) -> Path:
        """Persist the fitted feature engineer with joblib."""
        self._ensure_fitted()
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, output_path)
        return output_path.resolve()

    @classmethod
    def load(cls, path: Path | str) -> FeatureEngineer:
        """Load a persisted feature engineer."""
        artifact_path = Path(path)
        if not artifact_path.is_file():
            raise FileNotFoundError(
                f"Feature engineer artifact not found: {artifact_path}"
            )
        engineer = joblib.load(artifact_path)
        if not isinstance(engineer, cls):
            raise TypeError(
                f"Expected FeatureEngineer artifact, received {type(engineer)}"
            )
        return engineer


def _load_processed_parquet(path: Path) -> pd.DataFrame:
    """Load a processed parquet dataset."""
    if not path.is_file():
        raise FileNotFoundError(f"Processed dataset not found: {path}")
    return pd.read_parquet(path)


def _save_parquet(df: pd.DataFrame, path: Path) -> Path:
    """Save a dataframe to parquet."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return path.resolve()


@timer
def main() -> None:
    """Run feature engineering on processed datasets."""
    config = Config()
    ensure_directories(config)
    setup_logging(config)
    set_random_seed(config.random_seed)

    processed_train_path = config.processed_data_dir / "train_processed.parquet"
    processed_test_path = config.processed_data_dir / "test_processed.parquet"

    print_section("Loading datasets...")
    train_processed = _load_processed_parquet(processed_train_path)
    test_processed = _load_processed_parquet(processed_test_path)

    raw_train = load_train(config)
    raw_test = load_test(config)
    raw_train, cleanup_report = remove_missing_target(
        raw_train,
        config.target_column,
    )
    logger.info(
        "Removed %s rows with missing target from raw train",
        cleanup_report["removed_row_count"],
    )

    print_section("Fitting feature engineering...")
    engineer = FeatureEngineer(
        target_column=config.target_column,
        id_column=config.id_column,
    )
    engineer.fit(train_processed, raw_df=raw_train)

    print_section("Transforming datasets...")
    train_features = engineer.transform(
        train_processed,
        raw_df=raw_train,
        dataset_name="train",
    )
    test_features = engineer.transform(
        test_processed,
        raw_df=raw_test,
        dataset_name="test",
    )

    print_section("Saving outputs...")
    train_path = config.processed_data_dir / "train_features.parquet"
    test_path = config.processed_data_dir / "test_features.parquet"
    model_path = config.models_dir / "feature_engineer.joblib"

    _save_parquet(train_features, train_path)
    _save_parquet(test_features, test_path)
    engineer.save(model_path)

    logger.info("Saved train features to %s", train_path)
    logger.info("Saved test features to %s", test_path)
    logger.info("Saved feature engineer to %s", model_path)
    print_section("Completed successfully.")


if __name__ == "__main__":
    import sys

    feature_engineering_module = ModuleType("feature_engineering")
    feature_engineering_module.__dict__.update(globals())
    sys.modules["feature_engineering"] = feature_engineering_module
    for _pickle_class in (FeatureEngineer, FeatureEngineeringConfig, QuantileBinEdges):
        _pickle_class.__module__ = "feature_engineering"
    main()

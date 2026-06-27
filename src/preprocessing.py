"""Reusable preprocessing pipeline for NHANES Age Prediction."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Literal, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    MinMaxScaler,
    OneHotEncoder,
    OrdinalEncoder,
    RobustScaler,
    StandardScaler,
)

from project_setup import (
    Config,
    ensure_directories,
    load_test,
    load_train,
    print_section,
    remove_missing_target,
    restore_id_column,
    set_random_seed,
    setup_logging,
    timer,
)

logger = logging.getLogger("nhanes")

NumericImputationStrategy = Literal["median", "mean", "knn"]
CategoricalImputationStrategy = Literal["most_frequent", "constant"]
EncodingStrategy = Literal["ordinal", "onehot"]
ScalingStrategy = Literal["none", "standard", "robust", "minmax"]


@dataclass
class PreprocessingConfig:
    """Configuration options for ``PreprocessingPipeline``."""

    target_column: str = "age_group"
    id_column: str = "SEQN"
    numeric_imputation: NumericImputationStrategy = "median"
    categorical_imputation: CategoricalImputationStrategy = "most_frequent"
    categorical_constant_value: str = "missing"
    encoding: EncodingStrategy = "ordinal"
    scaling: ScalingStrategy = "none"
    knn_imputer_n_neighbors: int = 5
    random_state: int = 42
    binary_unique_threshold: int = 2


@dataclass
class ColumnGroups:
    """Detected feature column groups."""

    numeric: list[str] = field(default_factory=list)
    categorical: list[str] = field(default_factory=list)
    binary: list[str] = field(default_factory=list)
    input_order: list[str] = field(default_factory=list)


class PreprocessingPipeline:
    """Fit-transform preprocessing pipeline with persistence support."""

    def __init__(self, config: PreprocessingConfig | None = None, **kwargs: Any) -> None:
        """Initialize the preprocessing pipeline.

        Args:
            config: Optional preprocessing configuration object.
            **kwargs: Keyword arguments forwarded to ``PreprocessingConfig``.
        """
        self.config = config if config is not None else PreprocessingConfig(**kwargs)
        self.column_groups_: ColumnGroups | None = None
        self.feature_names_out_: list[str] = []
        self._column_transformer: ColumnTransformer | None = None
        self.is_fitted_: bool = False

    @property
    def excluded_columns(self) -> set[str]:
        """Return columns excluded from feature preprocessing."""
        return {self.config.target_column, self.config.id_column}

    def _extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return feature columns, excluding target and identifier columns."""
        self.check_duplicate_columns(df)
        available_exclusions = [
            column for column in self.excluded_columns if column in df.columns
        ]
        features = df.drop(columns=available_exclusions).copy()
        if features.empty:
            raise ValueError("No feature columns available after exclusions")
        return features

    @staticmethod
    def check_missing_columns(
        df: pd.DataFrame,
        expected_columns: Sequence[str],
        dataset_name: str = "dataset",
    ) -> None:
        """Raise when expected columns are absent.

        Args:
            df: Dataframe to validate.
            expected_columns: Required column names.
            dataset_name: Human-readable dataset identifier.

        Raises:
            ValueError: If one or more expected columns are missing.
        """
        missing = sorted(set(expected_columns) - set(df.columns))
        if missing:
            raise ValueError(
                f"{dataset_name} is missing required columns: {missing}"
            )

    @staticmethod
    def check_duplicate_columns(
        df: pd.DataFrame,
        dataset_name: str = "dataset",
    ) -> None:
        """Raise when duplicate column names are present.

        Args:
            df: Dataframe to validate.
            dataset_name: Human-readable dataset identifier.

        Raises:
            ValueError: If duplicate column names exist.
        """
        duplicated = df.columns[df.columns.duplicated()].tolist()
        if duplicated:
            raise ValueError(
                f"{dataset_name} contains duplicate column names: {duplicated}"
            )

    @staticmethod
    def check_unsupported_dtypes(
        df: pd.DataFrame,
        dataset_name: str = "dataset",
    ) -> None:
        """Raise when unsupported column dtypes are detected.

        Args:
            df: Dataframe to validate.
            dataset_name: Human-readable dataset identifier.

        Raises:
            TypeError: If unsupported dtypes are present.
        """
        unsupported: list[str] = []
        for column in df.columns:
            series = df[column]
            if pd.api.types.is_numeric_dtype(series):
                continue
            if pd.api.types.is_bool_dtype(series):
                continue
            if pd.api.types.is_string_dtype(series) or series.dtype == object:
                continue
            if isinstance(series.dtype, pd.CategoricalDtype):
                continue
            unsupported.append(f"{column} ({series.dtype})")
        if unsupported:
            raise TypeError(
                f"{dataset_name} contains unsupported dtypes: {unsupported}"
            )

    def check_schema_mismatch(
        self,
        reference_columns: Sequence[str],
        df: pd.DataFrame,
        dataset_name: str = "dataset",
    ) -> None:
        """Raise when input schema differs from a reference schema.

        Args:
            reference_columns: Expected feature column names and order.
            df: Dataframe to validate.
            dataset_name: Human-readable dataset identifier.

        Raises:
            ValueError: If schema mismatch is detected.
        """
        features = self._extract_features(df)
        reference = list(reference_columns)
        current = list(features.columns)
        if current != reference:
            missing = sorted(set(reference) - set(current))
            extra = sorted(set(current) - set(reference))
            raise ValueError(
                f"{dataset_name} schema mismatch. "
                f"Missing columns: {missing}. Extra columns: {extra}."
            )

    def check_feature_mismatch(
        self,
        left: pd.DataFrame,
        right: pd.DataFrame,
        left_name: str = "left",
        right_name: str = "right",
    ) -> None:
        """Raise when processed feature columns differ between datasets.

        Args:
            left: First processed dataframe.
            right: Second processed dataframe.
            left_name: Identifier for the first dataset.
            right_name: Identifier for the second dataset.

        Raises:
            ValueError: If feature columns or dtypes differ.
        """
        left_features = self._extract_features(left)
        right_features = self._extract_features(right)
        if list(left_features.columns) != list(right_features.columns):
            raise ValueError(
                f"Feature mismatch between {left_name} and {right_name}: "
                f"{list(left_features.columns)} != {list(right_features.columns)}"
            )
        dtype_mismatch = {
            column: (left_features[column].dtype, right_features[column].dtype)
            for column in left_features.columns
            if left_features[column].dtype != right_features[column].dtype
        }
        if dtype_mismatch:
            raise ValueError(
                f"Dtype mismatch between {left_name} and {right_name}: "
                f"{dtype_mismatch}"
            )

    def verify_processed_schemas_match(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
    ) -> None:
        """Verify processed train and test feature schemas are aligned.

        Args:
            train_df: Processed training dataframe.
            test_df: Processed test dataframe.

        Raises:
            ValueError: If schemas do not match.
        """
        self.check_feature_mismatch(
            train_df,
            test_df,
            left_name="train",
            right_name="test",
        )
        if self.feature_names_out_:
            train_features = self._extract_features(train_df)
            if list(train_features.columns) != self.feature_names_out_:
                raise ValueError(
                    "Processed train features do not match fitted output schema"
                )

    def detect_column_groups(self, df: pd.DataFrame) -> ColumnGroups:
        """Detect numeric, categorical, and binary feature columns.

        Args:
            df: Feature dataframe.

        Returns:
            Detected column groups preserving input order.
        """
        numeric_columns: list[str] = []
        categorical_columns: list[str] = []
        binary_columns: list[str] = []

        for column in df.columns:
            series = df[column]
            unique_count = int(series.nunique(dropna=True))
            is_binary = unique_count <= self.config.binary_unique_threshold

            if is_binary:
                binary_columns.append(column)

            if pd.api.types.is_numeric_dtype(series) and not is_binary:
                numeric_columns.append(column)
            else:
                categorical_columns.append(column)

        return ColumnGroups(
            numeric=numeric_columns,
            categorical=categorical_columns,
            binary=binary_columns,
            input_order=list(df.columns),
        )

    def _build_numeric_pipeline(self) -> Pipeline:
        """Build the numeric imputation and scaling pipeline."""
        if self.config.numeric_imputation == "median":
            imputer: Any = SimpleImputer(strategy="median")
        elif self.config.numeric_imputation == "mean":
            imputer = SimpleImputer(strategy="mean")
        else:
            imputer = KNNImputer(
                n_neighbors=self.config.knn_imputer_n_neighbors,
            )

        steps: list[tuple[str, Any]] = [("imputer", imputer)]
        if self.config.scaling == "standard":
            steps.append(("scaler", StandardScaler()))
        elif self.config.scaling == "robust":
            steps.append(("scaler", RobustScaler()))
        elif self.config.scaling == "minmax":
            steps.append(("scaler", MinMaxScaler()))
        return Pipeline(steps)

    def _build_categorical_pipeline(self) -> Pipeline:
        """Build the categorical imputation and encoding pipeline."""
        if self.config.categorical_imputation == "most_frequent":
            imputer: Any = SimpleImputer(strategy="most_frequent")
        else:
            imputer = SimpleImputer(
                strategy="constant",
                fill_value=self.config.categorical_constant_value,
            )

        if self.config.encoding == "ordinal":
            encoder: Any = OrdinalEncoder(
                handle_unknown="use_encoded_value",
                unknown_value=-1,
                dtype=np.float64,
            )
        else:
            encoder = OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False,
            )

        return Pipeline(
            [
                ("imputer", imputer),
                ("encoder", encoder),
            ]
        )

    def _build_column_transformer(
        self,
        column_groups: ColumnGroups,
    ) -> ColumnTransformer:
        """Construct a column transformer from detected column groups."""
        transformers: list[tuple[str, Pipeline, list[str]]] = []

        if column_groups.numeric:
            transformers.append(
                (
                    "numeric",
                    self._build_numeric_pipeline(),
                    column_groups.numeric,
                )
            )
        if column_groups.categorical:
            transformers.append(
                (
                    "categorical",
                    self._build_categorical_pipeline(),
                    column_groups.categorical,
                )
            )
        if not transformers:
            raise ValueError("No columns available for preprocessing")

        return ColumnTransformer(
            transformers=transformers,
            remainder="drop",
            verbose_feature_names_out=False,
        )

    @staticmethod
    def _clean_feature_names(raw_names: Iterable[str]) -> list[str]:
        """Normalize column transformer output feature names."""
        cleaned: list[str] = []
        for name in raw_names:
            if "__" in name:
                cleaned.append(name.split("__", 1)[1])
            else:
                cleaned.append(name)
        return cleaned

    def _validate_input(self, df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
        """Run reusable validation checks before fit or transform."""
        self.check_duplicate_columns(df, dataset_name=dataset_name)
        self.check_unsupported_dtypes(df, dataset_name=dataset_name)
        return self._extract_features(df)

    def fit(self, df: pd.DataFrame) -> PreprocessingPipeline:
        """Fit preprocessing transformers on training features.

        Args:
            df: Training dataframe containing features and optional metadata.

        Returns:
            Fitted pipeline instance.
        """
        features = self._validate_input(df, dataset_name="train")
        self.column_groups_ = self.detect_column_groups(features)
        self._column_transformer = self._build_column_transformer(
            self.column_groups_
        )
        self._column_transformer.fit(features)
        raw_names = self._column_transformer.get_feature_names_out()
        self.feature_names_out_ = self._clean_feature_names(raw_names)
        self.is_fitted_ = True
        logger.info(
            "Fitted preprocessing on %s rows, %s output features",
            len(features),
            len(self.feature_names_out_),
        )
        return self

    def transform(self, df: pd.DataFrame, dataset_name: str = "dataset") -> pd.DataFrame:
        """Transform features using the fitted preprocessing pipeline.

        Args:
            df: Input dataframe containing features and optional metadata.
            dataset_name: Human-readable dataset identifier.

        Returns:
            Transformed feature dataframe with preserved row order.

        Raises:
            RuntimeError: If the pipeline has not been fitted.
            ValueError: If schema validation fails.
        """
        if not self.is_fitted_ or self._column_transformer is None:
            raise RuntimeError("PreprocessingPipeline must be fitted before transform")
        if self.column_groups_ is None:
            raise RuntimeError("Column groups were not initialized during fit")

        features = self._validate_input(df, dataset_name=dataset_name)
        self.check_schema_mismatch(
            self.column_groups_.input_order,
            df,
            dataset_name=dataset_name,
        )

        transformed = self._column_transformer.transform(features)
        output = pd.DataFrame(
            transformed,
            columns=self.feature_names_out_,
            index=features.index,
        )
        return output[self.feature_names_out_]

    def fit_transform(
        self,
        df: pd.DataFrame,
        dataset_name: str = "train",
    ) -> pd.DataFrame:
        """Fit the pipeline and transform the input dataframe.

        Args:
            df: Input dataframe containing features and optional metadata.
            dataset_name: Human-readable dataset identifier.

        Returns:
            Transformed feature dataframe.
        """
        self.fit(df)
        return self.transform(df, dataset_name=dataset_name)

    def save(self, path: Path | str) -> Path:
        """Persist the fitted pipeline with joblib.

        Args:
            path: Destination file path.

        Returns:
            Absolute path to the saved artifact.

        Raises:
            RuntimeError: If the pipeline has not been fitted.
        """
        if not self.is_fitted_:
            raise RuntimeError("Cannot save an unfitted PreprocessingPipeline")
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, output_path)
        return output_path.resolve()

    @classmethod
    def load(cls, path: Path | str) -> PreprocessingPipeline:
        """Load a persisted preprocessing pipeline.

        Args:
            path: Path to a joblib artifact.

        Returns:
            Loaded preprocessing pipeline.

        Raises:
            FileNotFoundError: If the artifact does not exist.
        """
        artifact_path = Path(path)
        if not artifact_path.is_file():
            raise FileNotFoundError(f"Preprocessing artifact not found: {artifact_path}")
        pipeline = joblib.load(artifact_path)
        if not isinstance(pipeline, cls):
            raise TypeError(
                f"Expected PreprocessingPipeline artifact, received {type(pipeline)}"
            )
        return pipeline


def _save_parquet(df: pd.DataFrame, path: Path) -> Path:
    """Save a dataframe to parquet, creating parent directories if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return path.resolve()


@timer
def main() -> None:
    """Fit preprocessing on train data and persist processed datasets."""
    config = Config()
    ensure_directories(config)
    setup_logging(config)
    set_random_seed(config.random_seed)

    print_section("Loading datasets...")
    train_df = load_train(config)
    test_df = load_test(config)
    train_df, cleanup_report = remove_missing_target(
        train_df,
        config.target_column,
    )
    logger.info("Removed %s rows with missing target", cleanup_report["removed_row_count"])

    print_section("Fitting preprocessing pipeline...")
    pipeline = PreprocessingPipeline(
        target_column=config.target_column,
        id_column=config.id_column,
        random_state=config.random_seed,
    )
    pipeline.fit(train_df)

    print_section("Transforming datasets...")
    train_ids = train_df[config.id_column].copy()
    test_ids = test_df[config.id_column].copy()
    train_target = train_df[config.target_column].copy()

    train_features = pipeline.transform(train_df, dataset_name="train")
    test_features = pipeline.transform(test_df, dataset_name="test")

    train_processed = restore_id_column(
        train_features,
        train_ids,
        config.id_column,
    )
    train_processed[config.target_column] = train_target.values
    test_processed = restore_id_column(
        test_features,
        test_ids,
        config.id_column,
    )

    print_section("Verifying schemas...")
    pipeline.verify_processed_schemas_match(train_processed, test_processed)

    print_section("Saving outputs...")
    train_path = config.processed_data_dir / "train_processed.parquet"
    test_path = config.processed_data_dir / "test_processed.parquet"
    model_path = config.models_dir / "preprocessing.joblib"

    _save_parquet(train_processed, train_path)
    _save_parquet(test_processed, test_path)
    pipeline.save(model_path)

    logger.info("Saved train processed data to %s", train_path)
    logger.info("Saved test processed data to %s", test_path)
    logger.info("Saved preprocessing pipeline to %s", model_path)
    print_section("Completed successfully.")


if __name__ == "__main__":
    import sys
    from types import ModuleType

    preprocessing_module = ModuleType("preprocessing")
    preprocessing_module.__dict__.update(globals())
    sys.modules["preprocessing"] = preprocessing_module
    for _pickle_class in (PreprocessingPipeline, PreprocessingConfig, ColumnGroups):
        _pickle_class.__module__ = "preprocessing"
    main()

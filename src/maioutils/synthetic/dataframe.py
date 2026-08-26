from __future__ import annotations

import math
import re
import string
import warnings
from dataclasses import dataclass
from typing import Hashable, Sequence

import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata


@dataclass
class SyntheticDataConfig:
    """Configuration for synthetic dataframe generation."""

    n_rows: int | None = None
    random_state: int | None = None

    # Column handling
    keep_column_names: bool = True
    generated_column_prefix: str = "column"

    # Identifier handling
    keep_ids: bool = False
    id_columns: Sequence[Hashable] | None = None
    detect_id_columns: bool = True

    # Statistical behavior
    preserve_numeric_correlations: bool = True
    preserve_missingness: bool = True
    preserve_joint_missingness: bool = True

    # Type-detection thresholds
    categorical_max_unique: int = 30
    categorical_max_unique_fraction: float = 0.05
    id_min_unique_fraction: float = 0.90

    # Privacy-related controls
    minimum_category_count: int = 1
    numeric_jitter: float = 0.01


class SyntheticDataGenerator:
    """
    Generate a fake dataframe resembling the schema and broad statistical
    characteristics of a source dataframe.

    Important
    ---------
    This is a prototyping utility, not a formal privacy mechanism. It does not
    provide differential privacy guarantees. Rare categories, exact ranges,
    and small datasets may still reveal information about the source data.
    """

    _ID_NAME_PATTERN = re.compile(
        r"(^id$|_id$|^id_|identifier|subject|participant|patient|person|"
        r"sample_id|record_id|mrn|uuid|guid|accession)",
        flags=re.IGNORECASE,
    )

    def __init__(self, config: SyntheticDataConfig | None = None):
        self.config = config or SyntheticDataConfig()
        self.rng = np.random.default_rng(self.config.random_state)

    def generate(self, dataframe: pd.DataFrame) -> pd.DataFrame:
	    """Generate a synthetic version of `dataframe`."""
	    if not isinstance(dataframe, pd.DataFrame):
	        raise TypeError("dataframe must be a pandas DataFrame.")
	
	    if dataframe.columns.has_duplicates:
	        raise ValueError(
	            "Duplicate column names are not supported. "
	            "Rename duplicated columns before generating synthetic data."
	        )
	
	    if len(dataframe) == 0:
	        return self._empty_like(dataframe)
	
	    n_rows = (
	        len(dataframe)
	        if self.config.n_rows is None
	        else int(self.config.n_rows)
	    )
	
	    if n_rows < 0:
	        raise ValueError("n_rows must be non-negative.")
	
	    column_types = {
	        column: self._detect_column_type(dataframe[column])
	        for column in dataframe.columns
	    }
	
	    id_columns = self._resolve_id_columns(
	        dataframe,
	        column_types,
	    )
	
	    numeric_columns = [
	        column
	        for column, detected_type in column_types.items()
	        if detected_type in {"integer", "float"}
	        and column not in id_columns
	    ]
	
	    generated_columns: dict[Hashable, pd.Series | np.ndarray] = {}
	
	    # Generate related numeric columns together.
	    if numeric_columns:
	        numeric_result = self._generate_numeric_block(
	            dataframe[numeric_columns],
	            n_rows=n_rows,
	            preserve_correlations=self.config.preserve_numeric_correlations,
	        )
	
	        for column in numeric_columns:
	            generated_columns[column] = numeric_result[column].to_numpy()
	
	    # Generate all remaining columns.
	    for column in dataframe.columns:
	        if column in numeric_columns:
	            continue
	
	        source = dataframe[column]
	
	        if column in id_columns:
	            generated_columns[column] = self._generate_id_column(
	                source,
	                n_rows=n_rows,
	                keep_original=self.config.keep_ids,
	            )
	            continue
	
	        detected_type = column_types[column]
	
	        if detected_type == "boolean":
	            values = self._generate_boolean(source, n_rows)
	
	        elif detected_type == "datetime":
	            values = self._generate_datetime(source, n_rows)
	
	        elif detected_type == "categorical":
	            values = self._generate_categorical(source, n_rows)
	
	        elif detected_type == "string":
	            values = self._generate_string_column(source, n_rows)
	
	        elif detected_type == "timedelta":
	            values = self._generate_timedelta(source, n_rows)
	
	        else:
	            values = self._generate_fallback(source, n_rows)
	
	        generated_columns[column] = values
	
	    # Construct once rather than repeatedly inserting columns.
	    synthetic = pd.DataFrame(
	        {
	            column: (
	                generated_columns[column].reset_index(drop=True)
	                if isinstance(generated_columns[column], pd.Series)
	                else generated_columns[column]
	            )
	            for column in dataframe.columns
	        },
	        index=pd.RangeIndex(n_rows),
	    )
	
	    if self.config.preserve_missingness:
	        synthetic = self._apply_missingness(
	            source=dataframe,
	            synthetic=synthetic,
	            id_columns=id_columns,
	        )
	
	    synthetic = self._restore_dtypes(
	        source=dataframe,
	        synthetic=synthetic,
	        column_types=column_types,
	    )
	
	    if not self.config.keep_column_names:
	        synthetic.columns = [
	            f"{self.config.generated_column_prefix}_{i:03d}"
	            for i in range(1, synthetic.shape[1] + 1)
	        ]
	
	    return synthetic

    def _detect_column_type(self, series: pd.Series) -> str:
        """Infer the broad semantic type of a column."""
        dtype = series.dtype

        if isinstance(dtype, pd.CategoricalDtype):
            return "categorical"

        if pd.api.types.is_bool_dtype(dtype):
            return "boolean"

        if pd.api.types.is_datetime64_any_dtype(dtype):
            return "datetime"

        if pd.api.types.is_timedelta64_dtype(dtype):
            return "timedelta"

        if pd.api.types.is_integer_dtype(dtype):
            return "integer"

        if pd.api.types.is_float_dtype(dtype):
            nonmissing = series.dropna()
            if len(nonmissing) > 0 and np.allclose(
                nonmissing.to_numpy(),
                np.round(nonmissing.to_numpy()),
            ):
                return "integer"
            return "float"

        nonmissing = series.dropna()
        if len(nonmissing) == 0:
            return "string"

        n_unique = nonmissing.nunique(dropna=True)
        unique_fraction = n_unique / len(nonmissing)

        if (
            n_unique <= self.config.categorical_max_unique
            or unique_fraction <= self.config.categorical_max_unique_fraction
        ):
            return "categorical"

        return "string"

    def _resolve_id_columns(
        self,
        dataframe: pd.DataFrame,
        column_types: dict[Hashable, str],
    ) -> set[Hashable]:
        """Combine explicit ID columns with heuristic ID detection."""
        id_columns: set[Hashable] = set(self.config.id_columns or [])

        unknown_columns = id_columns.difference(dataframe.columns)
        if unknown_columns:
            raise KeyError(
                f"ID columns not found in dataframe: {sorted(unknown_columns)}"
            )

        if not self.config.detect_id_columns:
            return id_columns

        for column in dataframe.columns:
            if column in id_columns:
                continue

            series = dataframe[column].dropna()
            if len(series) == 0:
                continue

            unique_fraction = series.nunique(dropna=True) / len(series)
            name_looks_like_id = bool(
                self._ID_NAME_PATTERN.search(str(column))
            )

            values_look_unique = (
                unique_fraction >= self.config.id_min_unique_fraction
            )

            # Avoid automatically treating ordinary continuous measurements
            # as IDs based only on uniqueness.
            type_can_be_id = column_types[column] in {
                "integer",
                "string",
                "categorical",
            }

            if type_can_be_id and name_looks_like_id and values_look_unique:
                id_columns.add(column)

        return id_columns

    def _generate_numeric_block(
	    self,
	    dataframe: pd.DataFrame,
	    n_rows: int,
	    preserve_correlations: bool,
	) -> pd.DataFrame:
	    """
	    Generate numeric variables using empirical marginal distributions.
	
	    When requested, a Gaussian copula approximates rank correlations among
	    the numeric columns.
	    """
	    generated_columns: dict[Hashable, np.ndarray] = {}
	
	    usable_columns = [
	        column
	        for column in dataframe.columns
	        if dataframe[column].dropna().nunique() > 1
	    ]
	
	    constant_columns = [
	        column
	        for column in dataframe.columns
	        if column not in usable_columns
	    ]
	
	    for column in constant_columns:
	        observed = dataframe[column].dropna()
	
	        if len(observed):
	            generated_columns[column] = np.repeat(
	                observed.iloc[0],
	                n_rows,
	            )
	        else:
	            generated_columns[column] = np.full(
	                n_rows,
	                np.nan,
	                dtype=float,
	            )
	
	    if usable_columns:
	        if preserve_correlations and len(usable_columns) > 1:
	            uniforms = self._sample_numeric_copula(
	                dataframe[usable_columns],
	                n_rows=n_rows,
	            )
	        else:
	            uniforms = self.rng.uniform(
	                low=0.0,
	                high=1.0,
	                size=(n_rows, len(usable_columns)),
	            )
	
	        for column_index, column in enumerate(usable_columns):
	            source = pd.to_numeric(
	                dataframe[column].dropna(),
	                errors="coerce",
	            ).dropna()
	
	            values = np.sort(source.to_numpy(dtype=float))
	            probabilities = uniforms[:, column_index]
	
	            generated = self._empirical_quantile(
	                values,
	                probabilities,
	            )
	
	            if self.config.numeric_jitter > 0 and len(values) > 2:
	                generated = self._add_small_numeric_jitter(
	                    generated,
	                    source_values=values,
	                )
	
	            minimum = np.nanmin(values)
	            maximum = np.nanmax(values)
	
	            generated = np.clip(
	                generated,
	                minimum,
	                maximum,
	            )
	
	            if self._is_integer_like(dataframe[column]):
	                generated = np.rint(generated)
	                generated = np.clip(
	                    generated,
	                    math.ceil(minimum),
	                    math.floor(maximum),
	                )
	
	            generated_columns[column] = generated
	
	    # Construct the dataframe only once to avoid fragmentation.
	    output = pd.DataFrame(
	        {
	            column: generated_columns[column]
	            for column in dataframe.columns
	        },
	        index=pd.RangeIndex(n_rows),
	    )
	
	    return output

    def _sample_numeric_copula(
        self,
        dataframe: pd.DataFrame,
        n_rows: int,
    ) -> np.ndarray:
        """Fit and sample a simple empirical Gaussian copula."""
        latent_columns = []

        for column in dataframe.columns:
            series = pd.to_numeric(dataframe[column], errors="coerce")

            # Median imputation is used only for estimating dependence.
            filled = series.fillna(series.median()).to_numpy(dtype=float)

            ranks = rankdata(filled, method="average")
            uniforms = (ranks - 0.5) / len(ranks)
            uniforms = np.clip(uniforms, 1e-6, 1 - 1e-6)

            latent_columns.append(norm.ppf(uniforms))

        latent = np.column_stack(latent_columns)

        correlation = np.corrcoef(latent, rowvar=False)
        correlation = np.atleast_2d(correlation)
        correlation = self._nearest_valid_correlation(correlation)

        sampled_latent = self.rng.multivariate_normal(
            mean=np.zeros(len(dataframe.columns)),
            cov=correlation,
            size=n_rows,
            check_valid="ignore",
        )

        return np.clip(norm.cdf(sampled_latent), 1e-6, 1 - 1e-6)

    @staticmethod
    def _nearest_valid_correlation(matrix: np.ndarray) -> np.ndarray:
        """Regularize a correlation matrix so it is positive semidefinite."""
        matrix = np.asarray(matrix, dtype=float)
        matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
        matrix = (matrix + matrix.T) / 2.0

        np.fill_diagonal(matrix, 1.0)

        eigenvalues, eigenvectors = np.linalg.eigh(matrix)
        eigenvalues = np.clip(eigenvalues, 1e-8, None)

        regularized = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T

        scale = np.sqrt(np.diag(regularized))
        regularized = regularized / np.outer(scale, scale)
        np.fill_diagonal(regularized, 1.0)

        return regularized

    @staticmethod
    def _empirical_quantile(
        sorted_values: np.ndarray,
        probabilities: np.ndarray,
    ) -> np.ndarray:
        """Map uniform probabilities through an empirical quantile function."""
        if len(sorted_values) == 1:
            return np.repeat(sorted_values[0], len(probabilities))

        positions = probabilities * (len(sorted_values) - 1)
        lower = np.floor(positions).astype(int)
        upper = np.ceil(positions).astype(int)
        fraction = positions - lower

        return (
            sorted_values[lower] * (1.0 - fraction)
            + sorted_values[upper] * fraction
        )

    def _add_small_numeric_jitter(
        self,
        values: np.ndarray,
        source_values: np.ndarray,
    ) -> np.ndarray:
        """Add limited noise to reduce exact reuse of observed numeric values."""
        unique_values = np.unique(source_values)

        if len(unique_values) < 2:
            return values

        differences = np.diff(unique_values)
        positive_differences = differences[differences > 0]

        if len(positive_differences) == 0:
            return values

        scale = np.median(positive_differences)
        noise_sd = scale * self.config.numeric_jitter

        return values + self.rng.normal(0.0, noise_sd, size=len(values))

    def _generate_categorical(
        self,
        series: pd.Series,
        n_rows: int,
    ) -> pd.Series:
        """Sample categories according to their observed frequencies."""
        observed = series.dropna()

        if len(observed) == 0:
            return pd.Series([pd.NA] * n_rows, index=pd.RangeIndex(n_rows))

        counts = observed.value_counts(dropna=True)

        if self.config.minimum_category_count > 1:
            rare = counts[counts < self.config.minimum_category_count].index
            observed = observed.mask(observed.isin(rare), "__OTHER__")
            counts = observed.value_counts(dropna=True)

        categories = counts.index.to_numpy(dtype=object)
        probabilities = counts.to_numpy(dtype=float)
        probabilities /= probabilities.sum()

        generated = self.rng.choice(
            categories,
            size=n_rows,
            replace=True,
            p=probabilities,
        )

        return pd.Series(generated, index=pd.RangeIndex(n_rows))

    def _generate_boolean(
        self,
        series: pd.Series,
        n_rows: int,
    ) -> pd.Series:
        """Generate Boolean values with the observed true/false frequency."""
        observed = series.dropna()

        if len(observed) == 0:
            return pd.Series([pd.NA] * n_rows, index=pd.RangeIndex(n_rows))

        probability_true = observed.astype(bool).mean()
        values = self.rng.random(n_rows) < probability_true

        return pd.Series(values, index=pd.RangeIndex(n_rows))

    def _generate_datetime(
        self,
        series: pd.Series,
        n_rows: int,
    ) -> pd.Series:
        """Generate datetimes from the empirical datetime distribution."""
        observed = pd.to_datetime(series.dropna(), errors="coerce").dropna()

        if len(observed) == 0:
            return pd.Series([pd.NaT] * n_rows, index=pd.RangeIndex(n_rows))

        integer_values = observed.astype("int64").to_numpy()
        probabilities = self.rng.uniform(size=n_rows)

        generated_ns = self._empirical_quantile(
            np.sort(integer_values),
            probabilities,
        ).astype("int64")

        return pd.Series(
            pd.to_datetime(generated_ns),
            index=pd.RangeIndex(n_rows),
        )

    def _generate_timedelta(
        self,
        series: pd.Series,
        n_rows: int,
    ) -> pd.Series:
        """Generate timedelta values from the empirical distribution."""
        observed = series.dropna()

        if len(observed) == 0:
            return pd.Series([pd.NaT] * n_rows, index=pd.RangeIndex(n_rows))

        integer_values = observed.astype("timedelta64[ns]").astype("int64")
        probabilities = self.rng.uniform(size=n_rows)

        generated_ns = self._empirical_quantile(
            np.sort(integer_values),
            probabilities,
        ).astype("int64")

        return pd.Series(
            pd.to_timedelta(generated_ns, unit="ns"),
            index=pd.RangeIndex(n_rows),
        )

    def _generate_string_column(
        self,
        series: pd.Series,
        n_rows: int,
    ) -> pd.Series:
        """
        Generate strings with lengths and character-position patterns modeled
        after observed values.
        """
        observed = series.dropna().astype(str)

        if len(observed) == 0:
            return pd.Series([pd.NA] * n_rows, index=pd.RangeIndex(n_rows))

        templates = self.rng.choice(
            observed.to_numpy(),
            size=n_rows,
            replace=True,
        )

        generated = [
            self._randomize_string_template(template)
            for template in templates
        ]

        return pd.Series(generated, index=pd.RangeIndex(n_rows))

    def _generate_id_column(
        self,
        series: pd.Series,
        n_rows: int,
        keep_original: bool,
    ) -> pd.Series:
        """Retain IDs or create unique synthetic IDs of a similar format."""
        observed = series.dropna()

        if keep_original:
            if n_rows == len(series):
                return series.reset_index(drop=True).copy()

            if n_rows <= len(observed):
                sampled = self.rng.choice(
                    observed.to_numpy(),
                    size=n_rows,
                    replace=False,
                )
            else:
                warnings.warn(
                    "More synthetic rows than available IDs. Original IDs "
                    "cannot remain unique, so additional synthetic IDs will "
                    "be generated.",
                    stacklevel=2,
                )
                original = observed.to_numpy()
                additional = self._generate_id_column(
                    series,
                    n_rows=n_rows - len(original),
                    keep_original=False,
                ).to_numpy()
                sampled = np.concatenate([original, additional])

            return pd.Series(sampled, index=pd.RangeIndex(n_rows))

        if len(observed) == 0:
            return pd.Series(
                [f"ID_{i:06d}" for i in range(1, n_rows + 1)],
                index=pd.RangeIndex(n_rows),
            )

        if self._is_integer_like(series):
            return self._generate_numeric_ids(observed, n_rows)

        return self._generate_string_ids(observed.astype(str), n_rows)

    def _generate_numeric_ids(
        self,
        observed: pd.Series,
        n_rows: int,
    ) -> pd.Series:
        """Generate unique integer IDs without reproducing source IDs."""
        original_values = {
            int(value)
            for value in pd.to_numeric(observed, errors="coerce").dropna()
        }

        minimum = min(original_values)
        maximum = max(original_values)
        span = max(maximum - minimum + 1, n_rows)

        # Start beyond the observed range to avoid accidentally reproducing IDs.
        start = maximum + max(1, span)
        generated = np.arange(start, start + n_rows, dtype=np.int64)

        return pd.Series(generated, index=pd.RangeIndex(n_rows))

    def _generate_string_ids(
        self,
        observed: pd.Series,
        n_rows: int,
    ) -> pd.Series:
        """Generate unique strings following observed character patterns."""
        original_values = set(observed)
        generated: list[str] = []
        generated_set: set[str] = set()

        templates = observed.to_numpy()
        max_attempts = max(1_000, n_rows * 50)

        attempts = 0
        while len(generated) < n_rows and attempts < max_attempts:
            template = str(self.rng.choice(templates))
            candidate = self._randomize_string_template(template)

            if candidate not in original_values and candidate not in generated_set:
                generated.append(candidate)
                generated_set.add(candidate)

            attempts += 1

        # Pattern space may be too small, such as IDs "A1", "A2", ...
        while len(generated) < n_rows:
            candidate = f"SYN_{self.rng.integers(0, 1 << 48):012X}"
            if candidate not in original_values and candidate not in generated_set:
                generated.append(candidate)
                generated_set.add(candidate)

        return pd.Series(generated, index=pd.RangeIndex(n_rows))

    def _randomize_string_template(self, template: str) -> str:
        """
        Replace each alphanumeric character with a random character of the
        same broad class while retaining punctuation and separators.
        """
        result = []

        for character in str(template):
            if character.isdigit():
                result.append(self.rng.choice(list(string.digits)))
            elif character.isupper():
                result.append(self.rng.choice(list(string.ascii_uppercase)))
            elif character.islower():
                result.append(self.rng.choice(list(string.ascii_lowercase)))
            elif character.isalpha():
                result.append(self.rng.choice(list(string.ascii_letters)))
            else:
                result.append(character)

        return "".join(result)

    def _apply_missingness(
	    self,
	    source: pd.DataFrame,
	    synthetic: pd.DataFrame,
	    id_columns: set[Hashable],
	) -> pd.DataFrame:
	    """Apply either joint or per-column missingness patterns."""
	    columns = list(source.columns)
	    n_rows = len(synthetic)
	
	    if self.config.preserve_joint_missingness and len(source) > 0:
	        sampled_rows = self.rng.integers(
	            low=0,
	            high=len(source),
	            size=n_rows,
	        )
	
	        missing_mask = (
	            source.isna()
	            .iloc[sampled_rows]
	            .reset_index(drop=True)
	        )
	
	    else:
	        mask_data = {}
	
	        for column in columns:
	            missing_fraction = source[column].isna().mean()
	            mask_data[column] = (
	                self.rng.random(n_rows) < missing_fraction
	            )
	
	        missing_mask = pd.DataFrame(
	            mask_data,
	            index=pd.RangeIndex(n_rows),
	        )
	
	    if self.config.keep_ids and len(source) == n_rows:
	        for column in id_columns:
	            missing_mask[column] = (
	                source[column]
	                .isna()
	                .reset_index(drop=True)
	            )
	
	    # Apply the entire mask at once.
	    result = synthetic.mask(missing_mask)
	
	    # Force a compact internal memory layout.
	    return result.copy()

    def _restore_dtypes(
        self,
        source: pd.DataFrame,
        synthetic: pd.DataFrame,
        column_types: dict[Hashable, str],
    ) -> pd.DataFrame:
        """Restore source-like pandas dtypes where safely possible."""
        result = synthetic.copy()

        for column in source.columns:
            source_dtype = source[column].dtype
            detected_type = column_types[column]

            try:
                if isinstance(source_dtype, pd.CategoricalDtype):
                    # Synthetic rare-category grouping may add __OTHER__.
                    original_categories = list(source_dtype.categories)
                    generated_categories = [
                        value
                        for value in result[column].dropna().unique()
                        if value not in original_categories
                    ]
                    categories = original_categories + generated_categories

                    result[column] = pd.Categorical(
                        result[column],
                        categories=categories,
                        ordered=source_dtype.ordered,
                    )

                elif detected_type == "integer":
                    if result[column].isna().any():
                        result[column] = pd.to_numeric(
                            result[column],
                            errors="coerce",
                        ).round().astype("Int64")
                    else:
                        result[column] = pd.to_numeric(
                            result[column],
                            errors="raise",
                        ).round().astype(source_dtype)

                elif detected_type == "float":
                    result[column] = pd.to_numeric(
                        result[column],
                        errors="coerce",
                    ).astype(source_dtype)

                elif detected_type == "boolean":
                    if result[column].isna().any():
                        result[column] = result[column].astype("boolean")
                    else:
                        result[column] = result[column].astype(source_dtype)

                elif detected_type == "datetime":
                    result[column] = pd.to_datetime(
                        result[column],
                        errors="coerce",
                    )

                elif detected_type == "timedelta":
                    result[column] = pd.to_timedelta(
                        result[column],
                        errors="coerce",
                    )

                elif pd.api.types.is_string_dtype(source_dtype):
                    result[column] = result[column].astype(source_dtype)

            except (TypeError, ValueError):
                # Retaining a usable synthetic column is preferable to failing
                # because of an unusual extension or object dtype.
                pass

        return result

    def _generate_fallback(
        self,
        series: pd.Series,
        n_rows: int,
    ) -> pd.Series:
        """Fallback for unusual object or extension-array columns."""
        return self._generate_string_column(series.astype("string"), n_rows)

    def _empty_like(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        result = dataframe.copy()

        if not self.config.keep_column_names:
            result.columns = [
                f"{self.config.generated_column_prefix}_{i:03d}"
                for i in range(1, result.shape[1] + 1)
            ]

        return result

    @staticmethod
    def _is_integer_like(series: pd.Series) -> bool:
        if pd.api.types.is_integer_dtype(series.dtype):
            return True

        numeric = pd.to_numeric(series.dropna(), errors="coerce").dropna()

        return (
            len(numeric) > 0
            and np.allclose(numeric.to_numpy(), np.round(numeric.to_numpy()))
        )


def make_synthetic_dataframe(
    dataframe: pd.DataFrame,
    *,
    n_rows: int | None = None,
    random_state: int | None = None,
    keep_column_names: bool = True,
    keep_ids: bool = False,
    id_columns: Sequence[Hashable] | None = None,
    id_in_index: bool = False,
    keep_index_name: bool = True,
    detect_id_columns: bool = True,
    preserve_numeric_correlations: bool = True,
    preserve_missingness: bool = True,
    preserve_joint_missingness: bool = True,
    minimum_category_count: int = 1,
    numeric_jitter: float = 0.01,
) -> pd.DataFrame:
    """
    Generate a synthetic dataframe resembling the schema and broad
    statistical characteristics of a source dataframe.

    The generated dataframe preserves, approximately:

    - Column data types.
    - Numeric ranges and marginal distributions.
    - Correlations among numeric columns.
    - Categorical value frequencies.
    - Missing-value frequencies and, optionally, joint missingness patterns.
    - String lengths and character formats.
    - Identifier formats.

    Parameters
    ----------
    dataframe:
        Source dataframe whose schema and broad statistical properties should
        be approximated.

    n_rows:
        Number of synthetic rows to generate. By default, the number of rows
        in the source dataframe is used.

    random_state:
        Seed used by the random-number generator. Set this to an integer to
        make the generated dataframe reproducible.

    keep_column_names:
        Whether to retain the original column names.

        When False, columns are renamed sequentially as ``column_001``,
        ``column_002``, and so on.

    keep_ids:
        Whether to retain the original identifier values.

        When False, new synthetic identifiers are generated. Their broad
        format is inferred from the original IDs. For example, numeric IDs
        produce numeric synthetic IDs, while strings retain similar lengths,
        separators, and character classes.

        Retaining original IDs is generally not advisable when the generated
        data will leave the private environment.

    id_columns:
        Explicit list of columns containing identifiers.

        Explicitly specifying identifier columns is more reliable than
        automatic detection. These columns are passed through the dedicated
        identifier-generation logic.

        This argument applies to ordinary dataframe columns. To treat the
        dataframe index as an identifier, use ``id_in_index=True``.

    id_in_index:
        Whether the dataframe index contains identifiers that should be
        processed by the identifier-generation logic.

        When True, the index is temporarily converted into a column, treated
        as an identifier, and then restored as the index of the synthetic
        dataframe.

        If ``keep_ids=False``, new unique index values are generated. Numeric
        indexes produce numeric IDs, while string indexes produce IDs with a
        similar character format.

        If ``keep_ids=True``, the original index values are retained when the
        output has the same number of rows.

        This option currently supports a standard single-level index. A
        pandas ``MultiIndex`` is not supported.

    keep_index_name:
        Whether to retain the original index name when ``id_in_index=True``.

        When False, the returned dataframe index has no name. This argument
        has no effect when ``id_in_index=False``.

    detect_id_columns:
        Whether to automatically detect identifier columns.

        Detection is based on both the column name and the proportion of
        unique values. Names such as ``subject_id``, ``patient_id``,
        ``sample_id``, and ``record_id`` are considered likely identifiers.

        Automatic detection does not inspect the dataframe index; use
        ``id_in_index=True`` for an index containing identifiers.

    preserve_numeric_correlations:
        Whether to approximate dependence among numeric variables using an
        empirical Gaussian copula.

        When False, numeric columns are generated independently from their
        marginal empirical distributions.

    preserve_missingness:
        Whether to reproduce missing-value frequencies from the source data.

    preserve_joint_missingness:
        Whether to preserve relationships among missing values by sampling
        complete missingness patterns from source rows.

        When False, missing values are generated independently for each
        column according to its observed missing-value fraction.

        This argument has no effect when ``preserve_missingness=False``.

    minimum_category_count:
        Categories occurring fewer than this number of times are replaced
        with ``"__OTHER__"`` before synthetic values are generated.

        Increasing this value can prevent rare categories from being exposed
        directly in the synthetic dataframe.

    numeric_jitter:
        Amount of small random noise added to generated continuous numeric
        values, relative to the typical spacing among source values.

        This reduces exact reproduction of observed numeric values. Set to
        zero to disable numeric jitter.

    Returns
    -------
    pd.DataFrame
        A synthetic dataframe with the requested number of rows.

        When ``id_in_index=True``, the returned dataframe uses the generated
        or retained identifiers as its index.

    Raises
    ------
    TypeError
        If ``dataframe`` is not a pandas DataFrame.

    ValueError
        If the dataframe contains duplicate column names, ``n_rows`` is
        negative, or ``id_in_index=True`` is used with a MultiIndex.

    KeyError
        If a column listed in ``id_columns`` is not present in the dataframe.

    Notes
    -----
    This function is intended to produce structurally realistic data for code
    development and testing. It does not provide formal privacy guarantees
    such as differential privacy.

    Care should be taken with small datasets, rare categories, extreme values,
    and unusual identifier formats, because these may still expose information
    about the source dataset.

    Examples
    --------
    Generate a synthetic dataframe with explicit ID columns:

    >>> fake_df = make_synthetic_dataframe(
    ...     real_df,
    ...     n_rows=5000,
    ...     random_state=42,
    ...     id_columns=["subject_id", "sample_id"],
    ...     keep_ids=False,
    ... )

    Generate new subject IDs when the subject ID is stored in the index:

    >>> fake_df = make_synthetic_dataframe(
    ...     real_df,
    ...     random_state=42,
    ...     id_in_index=True,
    ...     keep_ids=False,
    ... )

    Retain the original index IDs:

    >>> fake_df = make_synthetic_dataframe(
    ...     real_df,
    ...     id_in_index=True,
    ...     keep_ids=True,
    ... )
    """
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("dataframe must be a pandas DataFrame.")

    if id_in_index and isinstance(dataframe.index, pd.MultiIndex):
        raise ValueError(
            "id_in_index=True currently supports only a single-level index. "
            "Reset the MultiIndex into separate columns and list those columns "
            "in id_columns."
        )

    source = dataframe.copy()
    source_index_name = dataframe.index.name
    temporary_index_column = None

    if id_in_index:
        temporary_index_column = source_index_name or "__index_id__"

        # Prevent a collision if the dataframe already contains a column with
        # the same name as the index.
        while temporary_index_column in source.columns:
            temporary_index_column = f"_{temporary_index_column}"

        source = source.reset_index(names=temporary_index_column)

        resolved_id_columns = list(id_columns or [])
        resolved_id_columns.append(temporary_index_column)
    else:
        resolved_id_columns = id_columns

    config = SyntheticDataConfig(
        n_rows=n_rows,
        random_state=random_state,
        # Keep the temporary index column addressable until it is restored.
        # Any requested renaming is applied to the remaining columns below.
        keep_column_names=True if id_in_index else keep_column_names,
        keep_ids=keep_ids,
        id_columns=resolved_id_columns,
        detect_id_columns=detect_id_columns,
        preserve_numeric_correlations=preserve_numeric_correlations,
        preserve_missingness=preserve_missingness,
        preserve_joint_missingness=preserve_joint_missingness,
        minimum_category_count=minimum_category_count,
        numeric_jitter=numeric_jitter,
    )

    result = SyntheticDataGenerator(config).generate(source)

    if id_in_index:
        # Renaming columns before restoring the index would remove the
        # temporary index-column name.
        if not keep_column_names:
            non_index_columns = [
                column
                for column in result.columns
                if column != temporary_index_column
            ]

            rename_mapping = {
                column: f"{config.generated_column_prefix}_{i:03d}"
                for i, column in enumerate(non_index_columns, start=1)
            }
            result = result.rename(columns=rename_mapping)

        result = result.set_index(temporary_index_column)

        if keep_index_name:
            result.index.name = source_index_name
        else:
            result.index.name = None

    return result

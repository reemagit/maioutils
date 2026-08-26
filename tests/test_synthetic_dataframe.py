import numpy as np
import pandas as pd
import pytest

from maioutils.synthetic import make_synthetic_dataframe


@pytest.fixture
def source_dataframe():
    return pd.DataFrame(
        {
            "subject_id": [f"SUB-{i:03d}" for i in range(1, 41)],
            "age": pd.Series(range(20, 60), dtype="int64"),
            "score": np.linspace(1.5, 9.5, 40),
            "group": pd.Categorical(["control", "treated"] * 20),
            "measurement": pd.Series(
                [pd.NA if i % 4 == 0 else i for i in range(40)],
                dtype="Int64",
            ),
        }
    )


def test_public_import_paths_expose_the_same_function():
    from DataFaker import make_synthetic_dataframe as legacy
    from maioutils import make_synthetic_dataframe as top_level
    from maioutils.synthetic import make_synthetic_dataframe as synthetic

    assert top_level is synthetic
    assert legacy is synthetic


def test_basic_generation_preserves_columns_and_requested_size(source_dataframe):
    synthetic = make_synthetic_dataframe(
        source_dataframe,
        n_rows=75,
        random_state=42,
    )

    assert synthetic.shape == (75, len(source_dataframe.columns))
    assert list(synthetic.columns) == list(source_dataframe.columns)


def test_random_state_is_reproducible(source_dataframe):
    first = make_synthetic_dataframe(source_dataframe, random_state=17)
    second = make_synthetic_dataframe(source_dataframe, random_state=17)

    pd.testing.assert_frame_equal(first, second)


def test_integer_columns_remain_integer_like(source_dataframe):
    synthetic = make_synthetic_dataframe(source_dataframe, random_state=3)

    assert pd.api.types.is_integer_dtype(synthetic["age"].dtype)
    assert pd.api.types.is_integer_dtype(synthetic["measurement"].dtype)
    nonmissing = synthetic["measurement"].dropna().to_numpy(dtype=float)
    assert np.allclose(nonmissing, np.round(nonmissing))


def test_missing_values_are_generated_when_requested(source_dataframe):
    synthetic = make_synthetic_dataframe(
        source_dataframe,
        n_rows=200,
        random_state=9,
        preserve_missingness=True,
    )

    assert synthetic["measurement"].isna().any()
    assert not synthetic["age"].isna().any()


def test_categorical_column_retains_categorical_behavior(source_dataframe):
    synthetic = make_synthetic_dataframe(
        source_dataframe,
        n_rows=100,
        random_state=11,
    )

    assert isinstance(synthetic["group"].dtype, pd.CategoricalDtype)
    assert set(synthetic["group"].dropna()) <= {"control", "treated"}
    assert set(synthetic["group"].dropna()) == {"control", "treated"}


def test_explicit_id_column_generates_unique_non_source_ids():
    source = pd.DataFrame(
        {
            "code": [f"P-{i:04d}" for i in range(20)],
            "value": range(20),
        }
    )

    synthetic = make_synthetic_dataframe(
        source,
        n_rows=50,
        id_columns=["code"],
        detect_id_columns=False,
        keep_ids=False,
        random_state=5,
    )

    assert synthetic["code"].is_unique
    assert set(synthetic["code"]).isdisjoint(source["code"])


def test_string_id_fallback_is_seeded_and_reproducible():
    source = pd.DataFrame({"code": ["0"], "value": [1]})
    options = {
        "n_rows": 12,
        "id_columns": ["code"],
        "detect_id_columns": False,
        "keep_ids": False,
        "random_state": 31,
    }

    first = make_synthetic_dataframe(source, **options)
    second = make_synthetic_dataframe(source, **options)

    pd.testing.assert_frame_equal(first, second)
    assert first["code"].is_unique
    assert set(first["code"]).isdisjoint(source["code"])


def test_id_in_index_generates_unique_non_source_ids():
    source = pd.DataFrame(
        {"value": range(12)},
        index=pd.Index([f"ROW-{i:03d}" for i in range(12)], name="record_id"),
    )

    synthetic = make_synthetic_dataframe(
        source,
        n_rows=30,
        id_in_index=True,
        keep_ids=False,
        random_state=22,
    )

    assert list(synthetic.columns) == ["value"]
    assert synthetic.index.name == "record_id"
    assert synthetic.index.is_unique
    assert set(synthetic.index).isdisjoint(source.index)


def test_index_ids_work_when_columns_are_renamed():
    source = pd.DataFrame(
        {"value": range(8), "label": ["a", "b"] * 4},
        index=pd.Index(range(100, 108), name="row_id"),
    )

    synthetic = make_synthetic_dataframe(
        source,
        id_in_index=True,
        keep_column_names=False,
        random_state=8,
    )

    assert list(synthetic.columns) == ["column_001", "column_002"]
    assert synthetic.index.name == "row_id"
    assert synthetic.index.is_unique

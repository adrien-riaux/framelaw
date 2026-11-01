"""Tests for user-defined expression and batch-mask checks."""

import framelaw as fl
import polars as pl
import pytest


def _frame() -> pl.DataFrame:
    """Frame with one valid row and two failing rows for the sum predicate."""
    return pl.DataFrame(
        {
            "a": [1, 2, 3, 4],
            "b": [1, 2, 2, 4],
            "c": [2, 4, 6, 9],
        }
    )


def _schema(*checks: fl.DataFrameCheck) -> fl.Schema:
    """Int64 schema over the three columns, with the given table checks."""
    return fl.Schema(
        {
            "a": fl.Column(pl.Int64),
            "b": fl.Column(pl.Int64),
            "c": fl.Column(pl.Int64),
        },
        dataframe_checks=list(checks),
    )


def test_expression_check_reports_failing_rows() -> None:
    """An expression check fails only where its Boolean mask is false."""
    schema = _schema(
        fl.checks.table.check_expr(pl.col("a") + pl.col("b") == pl.col("c"), name="sum_ok")
    )

    report = schema.validate(_frame(), raise_on_error=False)

    assert report is not None
    assert len(report.errors) == 1
    error = report.errors[0]
    assert error.check == "sum_ok"
    assert error.rows == [2, 3]
    assert error.columns == ["a", "b", "c"]


def test_expression_check_accepts_a_callable() -> None:
    """A zero-argument callable can defer expression construction."""
    schema = _schema(fl.checks.table.check_expr(lambda: pl.col("a") >= 3, name="a_ge_3"))

    report = schema.validate(_frame(), raise_on_error=False)

    assert report is not None
    assert report.errors[0].rows == [0, 1]


def test_expression_check_rejects_non_boolean_results() -> None:
    """Expressions must produce a Boolean column."""
    schema = _schema(fl.checks.table.check_expr(pl.col("a") + 1, name="bad"))

    with pytest.raises(TypeError, match="must return Boolean"):
        schema.validate(_frame(), raise_on_error=False)


def test_expression_check_rejects_multiple_columns() -> None:
    """A single predicate column is required."""
    with pytest.raises(ValueError, match="exactly one column"):
        _schema(fl.checks.table.check_expr(pl.all(), name="bad")).validate(
            _frame(), raise_on_error=False
        )


def test_expression_check_rejects_aggregates() -> None:
    """Aggregating expressions cannot be used as row masks."""
    schema = _schema(fl.checks.table.check_expr(pl.col("a").max() > 0, name="agg"))

    with pytest.raises(ValueError, match="preserve the row count"):
        schema.validate(_frame(), raise_on_error=False)


def test_mask_check_accepts_a_polars_series() -> None:
    """A Boolean Series from the callback is used directly."""
    schema = _schema(
        fl.checks.table.check_mask(lambda frame: frame["a"] < 4, columns=["a"], name="a_lt_4")
    )

    report = schema.validate(_frame(), raise_on_error=False)

    assert report is not None
    assert report.errors[0].rows == [3]
    assert report.errors[0].columns == ["a"]


def test_mask_check_accepts_a_python_sequence() -> None:
    """Plain lists of booleans are wrapped into a Polars Series."""
    schema = _schema(
        fl.checks.table.check_mask(
            lambda frame: [value < 3 for value in frame["a"].to_list()],
            name="a_lt_3",
        )
    )

    report = schema.validate(_frame(), raise_on_error=False)

    assert report is not None
    assert report.errors[0].rows == [2, 3]


def test_mask_check_skips_nulls() -> None:
    """Null mask entries skip the row rather than failing it."""
    schema = _schema(
        fl.checks.table.check_mask(lambda frame: pl.Series([True, None, False, None]), name="mixed")
    )

    report = schema.validate(_frame(), raise_on_error=False)

    assert report is not None
    assert report.errors[0].rows == [2]


def test_mask_check_rejects_wrong_length() -> None:
    """The mask must cover the current chunk exactly."""
    schema = _schema(fl.checks.table.check_mask(lambda frame: pl.Series([True]), name="short"))

    with pytest.raises(ValueError, match="must return 4 value"):
        schema.validate(_frame(), raise_on_error=False)


def test_mask_check_rejects_non_boolean() -> None:
    """Mask values must be Boolean, not integers."""
    schema = _schema(fl.checks.table.check_mask(lambda frame: pl.Series([1, 0, 1, 0]), name="ints"))

    with pytest.raises(TypeError, match="must return Boolean"):
        schema.validate(_frame(), raise_on_error=False)


def test_mask_check_rejects_none() -> None:
    """Returning None is a contract violation, not a pass."""
    schema = _schema(fl.checks.table.check_mask(lambda frame: None, name="none"))

    with pytest.raises(TypeError, match="must return a Boolean sequence"):
        schema.validate(_frame(), raise_on_error=False)


def test_mask_check_requires_a_callable() -> None:
    """check_mask rejects non-callables at construction time."""
    with pytest.raises(TypeError, match="expects a callable"):
        fl.checks.table.check_mask(pl.col("a"), name="bad")


def test_custom_check_exceptions_propagate() -> None:
    """Bugs in user callbacks surface as Python exceptions."""

    def explode(frame: pl.DataFrame) -> pl.Series:
        raise RuntimeError("boom")

    schema = _schema(fl.checks.table.check_mask(explode, name="explode"))

    with pytest.raises(RuntimeError, match="boom"):
        schema.validate(_frame(), raise_on_error=False)


def test_chunked_custom_checks_run_per_chunk_and_offset_rows() -> None:
    """Chunked validation invokes callbacks once per batch with absolute rows."""
    calls: list[int] = []

    def track(frame: pl.DataFrame) -> pl.Series:
        calls.append(frame.height)
        return frame["a"] < 4

    schema = _schema(fl.checks.table.check_mask(track, name="a_lt_4"))

    report = schema.validate(_frame(), raise_on_error=False, chunk_rows=2)

    assert report is not None
    assert calls == [2, 2]
    assert report.errors[0].rows == [3]


def test_custom_checks_respect_lazy_mode() -> None:
    """A lazy schema returns custom-check failures instead of raising."""
    schema = fl.Schema(
        {"a": fl.Column(pl.Int64)},
        dataframe_checks=[fl.checks.table.check_expr(pl.col("a") > 10, name="big")],
        lazy=True,
    )

    report = schema.validate(pl.DataFrame({"a": [1, 20]}))

    assert report is not None
    assert report.errors[0].rows == [0]


def test_custom_checks_raise_when_not_lazy() -> None:
    """The default schema still raises on custom-check failures."""
    schema = fl.Schema(
        {"a": fl.Column(pl.Int64)},
        dataframe_checks=[fl.checks.table.check_expr(pl.col("a") > 10, name="big")],
    )

    with pytest.raises(ValueError, match="custom check 'big'"):
        schema.validate(pl.DataFrame({"a": [1, 20]}))


def test_json_serialization_is_refused() -> None:
    """Schemas with custom callables cannot be serialized."""
    schema = _schema(fl.checks.table.check_mask(lambda frame: frame["a"] > 0, name="positive"))

    with pytest.raises(ValueError, match="Cannot serialize"):
        schema.to_json()


def test_json_serialization_still_works_without_custom_checks() -> None:
    """Native-only schemas keep round-tripping through JSON."""
    schema = _schema(fl.checks.table.unique_together(["a", "b"]))

    assert fl.Schema.from_json(schema.to_json()) is not None


def test_sampling_ignores_custom_checks() -> None:
    """Data generation has no way to honor custom predicates."""
    schema = _schema(fl.checks.table.check_expr(pl.col("a") > 1000, name="impossible"))

    sample = schema.sample(5, seed=7)

    assert sample.height == 5


def test_max_reported_rows_truncates_the_report() -> None:
    """Failure indices are capped while the total count stays accurate."""
    schema = _schema(
        fl.checks.table.check_expr(pl.col("a") > 100, name="all_fail", max_reported_rows=2)
    )

    report = schema.validate(_frame(), raise_on_error=False)

    assert report is not None
    error = report.errors[0]
    assert error.rows == [0, 1]
    assert "4 row(s)" in error.message
    assert "first 2" in error.message


def test_custom_checks_on_empty_frames_pass() -> None:
    """A zero-row frame has nothing to fail."""
    empty = pl.DataFrame({"a": pl.Series([], dtype=pl.Int64), "b": pl.Series([], dtype=pl.Int64)})
    schema = fl.Schema(
        {"a": fl.Column(pl.Int64), "b": fl.Column(pl.Int64)},
        dataframe_checks=[fl.checks.table.check_expr(pl.col("a") == pl.col("b"))],
    )

    assert schema.validate(empty) is None


def test_custom_check_names_must_not_be_empty() -> None:
    """Empty names make reports unreadable and are rejected."""
    with pytest.raises(ValueError, match="name must not be empty"):
        fl.checks.table.check_expr(pl.col("a") > 0, name="")

    with pytest.raises(ValueError, match="name must not be empty"):
        fl.checks.table.check_mask(lambda frame: frame["a"] > 0, name="")


def test_max_reported_rows_must_be_positive() -> None:
    """A zero cap would silently drop every failure row."""
    with pytest.raises(ValueError, match="max_reported_rows"):
        fl.checks.table.check_expr(pl.col("a") > 0, max_reported_rows=0)


def test_check_expr_requires_an_expression() -> None:
    """Non-expression inputs are rejected at construction time."""
    with pytest.raises(TypeError, match="expects a polars.Expr"):
        fl.checks.table.check_expr(42, name="bad")


def test_row_check_reports_failing_rows() -> None:
    """A row check receives a dict and fails where it returns false."""
    schema = _schema(
        fl.checks.table.check_rows(lambda row: row["a"] + row["b"] == row["c"], name="sum_ok")
    )

    report = schema.validate(_frame(), raise_on_error=False)

    assert report is not None
    error = report.errors[0]
    assert error.check == "sum_ok"
    assert error.rows == [2, 3]
    assert error.columns == ["a", "b", "c"]


def test_row_check_receives_a_dict() -> None:
    """The callback sees one plain dict per row, not a Polars object."""
    seen: list[dict[str, int]] = []

    def track(row: dict[str, int]) -> bool:
        assert isinstance(row, dict)
        seen.append(row)
        return True

    schema = _schema(fl.checks.table.check_rows(track, name="track"))
    assert schema.validate(_frame()) is None
    assert seen == [
        {"a": 1, "b": 1, "c": 2},
        {"a": 2, "b": 2, "c": 4},
        {"a": 3, "b": 2, "c": 6},
        {"a": 4, "b": 4, "c": 9},
    ]


def test_row_check_respects_columns() -> None:
    """Selected columns shape both the input dict and error attribution."""
    seen: list[dict[str, int]] = []

    def track(row: dict[str, int]) -> bool:
        seen.append(row)
        return row["a"] < 4

    schema = _schema(fl.checks.table.check_rows(track, columns=["a"], name="a_lt_4"))

    report = schema.validate(_frame(), raise_on_error=False)

    assert report is not None
    assert report.errors[0].rows == [3]
    assert report.errors[0].columns == ["a"]
    assert seen == [{"a": 1}, {"a": 2}, {"a": 3}, {"a": 4}]


def test_row_check_skips_nulls() -> None:
    """Returning None skips the row instead of failing it."""
    schema = _schema(
        fl.checks.table.check_rows(
            lambda row: None if row["a"] == 2 else row["a"] < 4,
            columns=["a"],
            name="a_ok",
        )
    )

    report = schema.validate(_frame(), raise_on_error=False)

    assert report is not None
    assert report.errors[0].rows == [3]


def test_row_check_requires_a_callable() -> None:
    """check_rows rejects non-callables at construction time."""
    with pytest.raises(TypeError, match="expects a callable"):
        fl.checks.table.check_rows(42, name="bad")


def test_row_check_rejects_empty_columns() -> None:
    """An empty column selection would pass an empty dict to the callback."""
    with pytest.raises(ValueError, match="columns must not be empty"):
        fl.checks.table.check_rows(lambda row: True, columns=[], name="bad")


def test_row_check_exceptions_propagate() -> None:
    """Bugs in row callbacks surface as Python exceptions."""

    def explode(row: dict[str, int]) -> bool:
        raise RuntimeError("boom")

    schema = _schema(fl.checks.table.check_rows(explode, name="explode"))

    with pytest.raises(RuntimeError, match="boom"):
        schema.validate(_frame(), raise_on_error=False)


def test_row_check_rejects_non_boolean_results() -> None:
    """Row predicates must return booleans, not arbitrary truthy values."""
    schema = _schema(fl.checks.table.check_rows(lambda row: 1, name="bad"))

    with pytest.raises(pl.exceptions.SchemaError, match="Boolean"):
        schema.validate(_frame(), raise_on_error=False)


def test_row_check_defaults_to_custom_rows_name() -> None:
    """Unnamed row checks are reported as CustomRows with all columns."""
    schema = _schema(fl.checks.table.check_rows(lambda row: False))

    report = schema.validate(_frame(), raise_on_error=False)

    assert report is not None
    assert report.errors[0].check == "CustomRows"
    assert report.errors[0].columns == ["a", "b", "c"]


def test_row_check_runs_per_chunk_and_offsets_rows() -> None:
    """Chunked validation invokes row checks per batch with absolute rows."""
    schema = _schema(
        fl.checks.table.check_rows(lambda row: row["a"] < 4, columns=["a"], name="a_lt_4")
    )

    report = schema.validate(_frame(), raise_on_error=False, chunk_rows=2)

    assert report is not None
    assert report.errors[0].rows == [3]


def test_row_check_json_serialization_is_refused() -> None:
    """Schemas with row callables cannot be serialized."""
    schema = _schema(fl.checks.table.check_rows(lambda row: True, name="rows"))

    with pytest.raises(ValueError, match="Cannot serialize"):
        schema.to_json()


def test_lazy_chunked_validation_streams_batches() -> None:
    """Lazy inputs are streamed through collect_batches, one call per batch."""
    heights: list[int] = []

    def track(chunk: pl.DataFrame) -> pl.Series:
        heights.append(chunk.height)
        return chunk["a"] < 5

    frame = pl.DataFrame({"a": [1, 2, 3, 4, 5]})
    schema = fl.Schema(
        {"a": fl.Column(pl.Int64)},
        dataframe_checks=[fl.checks.table.check_mask(track, name="a_lt_5")],
    )

    report = schema.validate(frame.lazy(), raise_on_error=False, chunk_rows=2)

    assert report is not None
    assert heights == [2, 2, 1]
    assert report.errors[0].rows == [4]


def test_validate_lazy_accepts_chunk_rows() -> None:
    """validate_lazy can stream a LazyFrame without raising."""
    frame = pl.DataFrame({"a": [1, 2, 3]})
    schema = fl.Schema(
        {"a": fl.Column(pl.Int64)},
        dataframe_checks=[fl.checks.table.check_expr(pl.col("a") < 3, name="a_lt_3")],
    )

    report = schema.validate_lazy(frame.lazy(), chunk_rows=2)

    assert report is not None
    assert report.errors[0].rows == [2]

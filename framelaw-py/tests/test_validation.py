from datetime import date, datetime, timedelta
from importlib import metadata
from typing import Literal

import framelaw as fl
import polars as pl
import pytest
from pydantic import BaseModel, Field


def test_package_exposes_installed_version() -> None:
    """Expose the distribution version as ``framelaw.__version__``."""
    assert fl.__version__ == metadata.version("framelaw")


def test_validation_returns_report_with_rows() -> None:
    """Return structured failures when raising is disabled."""
    schema = fl.Schema(
        {
            "age": fl.Column(pl.Int64, nullable=False, checks=[fl.gt(0)]),
            "name": fl.Column(pl.String, nullable=False),
        }
    )
    dataframe = pl.DataFrame({"age": [1, -2, 3], "name": ["a", None, "c"]})

    report = schema.validate(dataframe, raise_on_error=False)

    assert not report.is_valid()
    assert any(error.column == "age" and error.rows == [1] for error in report.errors)
    assert any(error.column == "name" and error.rows == [1] for error in report.errors)
    with pytest.raises(ValueError, match="Validation failed"):
        schema.validate(dataframe)


def test_aliases_and_valid_data() -> None:
    """Expose the documented check aliases and accept valid data."""
    schema = fl.Schema(
        {
            "country": fl.Column(
                pl.String,
                checks=[fl.one_of(["FR", "US"]), fl.matches(r"^[A-Z]{2}$")],
            ),
            "id": fl.Column(pl.Int64, checks=[fl.unique()]),
        }
    )
    dataframe = pl.DataFrame({"country": ["FR", "US"], "id": [1, 2]})

    assert schema.validate(dataframe) is None


def test_invalid_regex_is_a_python_error() -> None:
    """Reject malformed regexes during check construction."""
    with pytest.raises(ValueError, match="Invalid regular expression"):
        fl.matches("[")


def test_string_checks_report_expected_rows() -> None:
    """Combine string membership with the new string checks."""
    schema = fl.Schema(
        {
            "code": fl.Column(
                pl.String,
                checks=[
                    fl.str_length_between(2, 5),
                    fl.not_blank(),
                    fl.starts_with("AB"),
                    fl.ends_with("-1"),
                    fl.is_in(["AB-1", "AB2-1"]),
                ],
            ),
        }
    )
    valid = pl.DataFrame({"code": ["AB-1", "AB2-1"]})
    assert schema.validate(valid) is None

    bad = pl.DataFrame({"code": ["ABCD-1", "", "cb-1", "AB-2", "AB-1x"]})
    report = schema.validate(bad, raise_on_error=False)
    assert not report.is_valid()
    rows_by_check = {e.check: e.rows for e in report.errors}
    assert rows_by_check["StrLengthBetween"] == [0, 1]
    assert rows_by_check["NotBlank"] == [1]
    assert rows_by_check["StartsWith"] == [1, 2]
    assert rows_by_check["EndsWith"] == [1, 3, 4]
    assert rows_by_check["OneOfString"] == [0, 1, 2, 3, 4]


def test_numeric_and_boolean_one_of_membership() -> None:
    """Generalize one_of membership to numbers and booleans."""
    schema = fl.Schema(
        {
            "level": fl.Column(pl.Int64, checks=[fl.is_in([2, 4, 6])]),
            "score": fl.Column(pl.Float64, checks=[fl.is_in([0.5, 1.5])]),
            "flag": fl.Column(pl.Boolean, checks=[fl.is_in([True])]),
        }
    )
    valid = pl.DataFrame({"level": [2, 4, 6], "score": [1.5, 0.5, 1.5], "flag": [True, True, True]})
    assert schema.validate(valid) is None

    bad = pl.DataFrame({"level": [3], "score": [1.0], "flag": [False]})
    report = schema.validate(bad, raise_on_error=False)
    assert all(e.check == "OneOfNumeric" for e in report.errors)
    assert len(report.errors) == 3
    assert next(e for e in report.errors if e.column == "level").rows == [0]
    assert next(e for e in report.errors if e.column == "score").rows == [0]
    assert next(e for e in report.errors if e.column == "flag").rows == [0]


def test_one_of_rejects_empty_and_mixed_lists() -> None:
    """Guard the generalized is_in constructor against bad inputs."""
    with pytest.raises(ValueError, match="empty"):
        fl.is_in([])
    with pytest.raises(ValueError, match="only strings or only numbers"):
        fl.is_in(["a", 1])
    with pytest.raises(ValueError, match="strings, numbers, or booleans"):
        fl.is_in([object()])
    with pytest.raises(ValueError, match="min_len must be <= max_len"):
        fl.str_length_between(5, 2)


def test_new_string_checks_json_round_trip() -> None:
    """Round-trip the new checks through schema JSON."""
    schema = fl.Schema(
        {
            "code": fl.Column(
                pl.String,
                checks=[fl.str_length_between(2, 4), fl.not_blank()],
            ),
            "file": fl.Column(pl.String, checks=[fl.starts_with("data."), fl.ends_with(".csv")]),
            "level": fl.Column(pl.Int64, checks=[fl.is_in([1, 3])]),
        }
    )
    payload = schema.to_json()
    loaded = fl.Schema.from_json(payload)
    frame = pl.DataFrame(
        {"code": ["ab", "abc"], "file": ["data.a.csv", "data.b.csv"], "level": [1, 3]}
    )
    assert loaded.validate(frame) is None
    bad = pl.DataFrame({"code": ["a", " "], "file": ["other.txt", "data.b.csv"], "level": [2, 4]})
    report = loaded.validate(bad, raise_on_error=False)
    assert not report.is_valid()
    assert next(
        e for e in report.errors if e.column == "code" and e.check == "StrLengthBetween"
    ).rows == [0, 1]
    assert next(e for e in report.errors if e.column == "code" and e.check == "NotBlank").rows == [
        1
    ]
    assert next(
        e for e in report.errors if e.column == "file" and e.check == "StartsWith"
    ).rows == [0]
    assert next(e for e in report.errors if e.column == "file" and e.check == "EndsWith").rows == [
        0
    ]
    assert next(
        e for e in report.errors if e.column == "level" and e.check == "OneOfNumeric"
    ).rows == [0, 1]


def test_new_string_checks_generation() -> None:
    """Sample valid data when schema uses the new string checks."""
    schema = fl.Schema(
        {
            "code": fl.Column(
                pl.String,
                checks=[fl.str_length_between(3, 8), fl.starts_with("x-"), fl.not_blank()],
            ),
            "level": fl.Column(pl.Int64, checks=[fl.is_in([1, 2])]),
        }
    )
    frame = fl.sample(schema, n=10, seed=7)
    assert schema.validate(frame) is None


def test_dtype_mismatch_is_reported() -> None:
    """Reject a DataFrame column whose native dtype differs from the schema."""
    schema = fl.Schema({"value": fl.Column(pl.Int32)})
    dataframe = pl.DataFrame({"value": [1, 2]})

    report = schema.validate(dataframe, raise_on_error=False)

    assert not report.is_valid()
    assert report.errors[0].check == "Dtype"
    assert report.errors[0].rows == []


def test_infer_schema_and_refine_column() -> None:
    """Infer a schema and add a stricter check through the column proxy."""
    dataframe = pl.DataFrame(
        {
            "age": [20, 30, 40],
            "email": ["a@example.com", "b@example.com", "c@example.com"],
        },
        schema={"age": pl.Int32, "email": pl.String},
    )

    schema = fl.infer_schema(dataframe)
    schema["age"].add_check(fl.lt(100))

    assert schema.validate(dataframe) is None
    with pytest.raises(KeyError):
        schema["missing"]

    invalid = dataframe.with_columns(
        pl.Series("age", [20, 30, 120], dtype=pl.Int32),
        pl.Series("email", ["a@example.com", "invalid", "c@example.com"]),
    )
    report = schema.validate(invalid, raise_on_error=False)
    assert any(error.column == "age" and error.check == "LessThanF64" for error in report.errors)
    assert any(
        error.column == "email" and error.check == "MatchesRegex" and error.rows == [1]
        for error in report.errors
    )


def test_infer_schema_temporal_bounds_and_sample() -> None:
    """Temporal columns infer tick bounds, and sample stays inside them."""
    from datetime import timedelta

    epoch = date(1970, 1, 1)
    dataframe = pl.DataFrame(
        {
            "start": [epoch, epoch + timedelta(days=50)],
            "end": [epoch + timedelta(days=1), epoch + timedelta(days=60)],
        },
        schema={"start": pl.Date, "end": pl.Date},
    )

    schema = fl.infer_schema(dataframe)
    assert schema.validate(dataframe) is None

    generated = fl.sample(schema, 40, seed=3).cast({"start": pl.Date, "end": pl.Date})
    assert schema.validate(generated) is None

    ticks = generated.select(
        (pl.col("start") - pl.lit(epoch)).dt.total_days().alias("start_tick"),
        (pl.col("end") - pl.lit(epoch)).dt.total_days().alias("end_tick"),
    )
    assert ticks["start_tick"].min() >= -1
    assert ticks["start_tick"].max() <= 50
    assert ticks["end_tick"].min() >= 0
    assert ticks["end_tick"].max() <= 61


def test_lazyframe_and_polars_schema_validation() -> None:
    """Collect LazyFrames once and build schemas from native Polars dtypes."""
    schema = fl.Schema.from_polars_schema({"value": pl.Int64})
    dataframe = pl.DataFrame({"value": [1, 2, 3]})

    assert schema.validate(dataframe.lazy()) is None


def test_report_to_dataframe_expands_failing_rows() -> None:
    """Represent each failing row as one native Polars report row."""
    schema = fl.Schema({"value": fl.Column(pl.Int64, nullable=False, checks=[fl.gt(0)])})
    dataframe = pl.DataFrame({"value": [1, -2, 0]})

    report = schema.validate(dataframe, raise_on_error=False)
    report_dataframe = report.to_dataframe()

    assert report_dataframe.columns == ["column", "columns", "check", "row", "message"]
    assert report_dataframe.select("row").to_series().to_list() == [1, 2]


def test_pydantic_scalar_constraints_and_optional_fields() -> None:
    """Map supported Pydantic v2 fields into native framelaw checks."""

    class User(BaseModel):
        age: int = Field(gt=0, lt=120)
        score: float = Field(ge=0.0, le=1.0)
        email: str = Field(pattern=r".+@.+")
        country: Literal["FR", "US"] | None = None
        nickname: str | None = None

    schema = fl.Schema.from_pydantic(User)
    valid = pl.DataFrame(
        {
            "age": [20],
            "score": [0.0],
            "email": ["user@example.com"],
            "country": ["FR"],
            "nickname": [None],
        }
    )
    assert schema.validate(valid) is None

    invalid = valid.with_columns(
        pl.Series("score", [1.5]),
        pl.Series("country", ["DE"]),
    )
    report = schema.validate(invalid, raise_on_error=False)
    assert any(error.column == "score" for error in report.errors)
    assert any(error.column == "country" for error in report.errors)


def test_pydantic_rejects_unsupported_fields() -> None:
    """Reject unsupported nested Pydantic types instead of silently weakening schemas."""

    class Nested(BaseModel):
        values: dict[str, int]

    with pytest.raises(TypeError, match="Unsupported Pydantic field type"):
        fl.Schema.from_pydantic(Nested)


def test_pydantic_list_fields_map_to_list_dtype() -> None:
    """Map list[T] fields into List dtypes with length checks."""

    class WithList(BaseModel):
        values: list[int] = Field(min_length=1, max_length=3)

    schema = fl.Schema.from_pydantic(WithList)
    valid = pl.DataFrame({"values": [[1, 2], [3]]})
    assert schema.validate(valid) is None

    invalid = pl.DataFrame({"values": [[], [1, 2, 3, 4]]})
    report = schema.validate(invalid, raise_on_error=False)
    assert any(error.column == "values" for error in report.errors)


def test_schema_json_round_trip_preserves_validation() -> None:
    """Round-trip supported dtypes, nullability, and every built-in check kind."""
    schema = fl.Schema(
        {
            "value": fl.Column(
                pl.Int64,
                nullable=False,
                checks=[
                    fl.not_null(),
                    fl.unique(),
                    fl.gt(0),
                    fl.lt(100),
                    fl.gte(1),
                    fl.lte(99),
                    fl.between(1, 99),
                ],
            ),
            "country": fl.Column(
                pl.String,
                checks=[fl.matches(r"^[A-Z]{2}$"), fl.one_of(["FR", "US"])],
            ),
        }
    )

    restored = fl.Schema.from_json(schema.to_json())
    dataframe = pl.DataFrame({"value": [1, 2], "country": ["FR", "US"]})

    assert restored.validate(dataframe) is None
    assert '"version":2' in schema.to_json()


def test_schema_json_rejects_invalid_versions_and_dtypes() -> None:
    """Reject unknown schema versions and unsupported persisted dtypes."""
    with pytest.raises(ValueError, match="Unsupported schema JSON version"):
        fl.Schema.from_json('{"version": 99, "columns": []}')

    with pytest.raises(ValueError, match="Unsupported JSON Polars dtype"):
        fl.Schema.from_json(
            '{"version": 2, "columns": [{"name": "x", "dtype": "Mystery", '
            '"nullable": true, "checks": []}]}'
        )


def test_schema_json_v1_still_loads() -> None:
    """Accept v1 payloads without table checks for backward compatibility."""
    restored = fl.Schema.from_json(
        '{"version": 1, "columns": [{"name": "value", "dtype": "Int64", '
        '"nullable": false, "checks": [{"type": "GreaterThan", "value": 0}]}]}'
    )
    assert restored.validate(pl.DataFrame({"value": [1, 2]})) is None


def test_temporal_dtypes_validate_and_round_trip() -> None:
    """Validate Date/Datetime columns and persist their temporal dtype tags."""
    schema = fl.Schema(
        {
            "day": fl.Column(pl.Date, checks=[fl.gt(0)]),
            "instant": fl.Column(pl.Datetime("us")),
        }
    )
    dataframe = pl.DataFrame(
        {
            "day": [date(1970, 1, 2), date(1970, 1, 3)],
            "instant": [datetime(2026, 1, 1), datetime(2026, 1, 2)],
        }
    )

    assert schema.validate(dataframe) is None
    restored = fl.Schema.from_json(schema.to_json())
    assert restored.validate(dataframe) is None


def test_cross_column_comparison_reports_rows() -> None:
    """Validate declarative comparisons between two columns."""
    schema = fl.Schema(
        {
            "start": fl.Column(pl.Int64),
            "end": fl.Column(pl.Int64),
        },
        dataframe_checks=[fl.lt_column("start", "end")],
    )
    valid = pl.DataFrame({"start": [1, 2], "end": [2, 3]})
    assert schema.validate(valid) is None

    invalid = pl.DataFrame({"start": [1, 5], "end": [2, 3]})
    report = schema.validate(invalid, raise_on_error=False)
    assert not report.is_valid()
    assert report.errors[0].columns == ["start", "end"]
    assert report.errors[0].column == "start"
    assert report.errors[0].rows == [1]


def test_unique_together_reports_duplicates() -> None:
    """Reject duplicated combinations across columns."""
    schema = fl.Schema(
        {
            "a": fl.Column(pl.String),
            "b": fl.Column(pl.Int64),
        },
        dataframe_checks=[fl.unique_together(["a", "b"])],
    )
    dataframe = pl.DataFrame({"a": ["x", "x", "y"], "b": [1, 1, 2]})
    report = schema.validate(dataframe, raise_on_error=False)
    assert not report.is_valid()
    assert report.errors[0].rows == [0, 1]


def test_cross_column_json_round_trip() -> None:
    """Persist table checks alongside column checks."""
    schema = fl.Schema(
        {"start": fl.Column(pl.Int64), "end": fl.Column(pl.Int64)},
        dataframe_checks=[fl.lt_column("start", "end")],
    )
    restored = fl.Schema.from_json(schema.to_json())
    assert restored.validate(pl.DataFrame({"start": [1], "end": [2]})) is None
    report = restored.validate(pl.DataFrame({"start": [5], "end": [2]}), raise_on_error=False)
    assert not report.is_valid()


def test_ordered_chain_reports_rows() -> None:
    """Validate the per-row chain `a <= b <= c` (and strict mode)."""
    schema = fl.Schema(
        {"a": fl.Column(pl.Int64), "b": fl.Column(pl.Int64), "c": fl.Column(pl.Int64)},
        dataframe_checks=[fl.ordered(["a", "b", "c"], strict=False)],
    )
    report = schema.validate(
        pl.DataFrame({"a": [1, 9, 2], "b": [2, 2, 4], "c": [3, 5, 7]}), raise_on_error=False
    )
    assert not report.is_valid()
    assert sorted({row for error in report.errors for row in error.rows}) == [1]

    strict = fl.Schema(
        {"a": fl.Column(pl.Int64), "b": fl.Column(pl.Int64), "c": fl.Column(pl.Int64)},
        dataframe_checks=[fl.ordered(["a", "b", "c"], strict=True)],
    )
    report = strict.validate(
        pl.DataFrame({"a": [1, 2, 2], "b": [2, 2, 3], "c": [3, 3, 4]}), raise_on_error=False
    )
    assert not report.is_valid()
    assert sorted({row for error in report.errors for row in error.rows}) == [1]


def test_sum_columns_reports_mismatched_rows() -> None:
    """Check `target == sum(addends)` exactly for ints, tolerantly for floats."""
    int_schema = fl.Schema(
        {"a": fl.Column(pl.Int64), "b": fl.Column(pl.Int64), "c": fl.Column(pl.Int64)},
        dataframe_checks=[fl.sum_columns(["a", "b"], "c")],
    )
    report = int_schema.validate(
        pl.DataFrame({"a": [1, 2], "b": [2, 3], "c": [3, 9]}), raise_on_error=False
    )
    assert not report.is_valid()
    assert report.errors[0].rows == [1]

    float_schema = fl.Schema(
        {"a": fl.Column(pl.Float64), "b": fl.Column(pl.Float64), "c": fl.Column(pl.Float64)},
        dataframe_checks=[fl.sum_columns(["a", "b"], "c", tolerance=1e-9)],
    )
    assert float_schema.validate(pl.DataFrame({"a": [0.1], "b": [0.2], "c": [0.3]})) is None
    report = float_schema.validate(
        pl.DataFrame({"a": [1.0], "b": [1.0], "c": [10.0]}), raise_on_error=False
    )
    assert not report.is_valid()
    assert report.errors[0].rows == [0]


def test_implies_reports_missing_values() -> None:
    """Non-null left values require a present `by` value."""
    schema = fl.Schema(
        {"left": fl.Column(pl.Int64), "by": fl.Column(pl.String)},
        dataframe_checks=[fl.implies("left", "by")],
    )
    report = schema.validate(
        pl.DataFrame({"left": [1, None, 3], "by": ["x", None, None]}), raise_on_error=False
    )
    assert not report.is_valid()
    assert report.errors[0].rows == [2]
    assert schema.validate(pl.DataFrame({"left": [1, None], "by": ["x", None]})) is None


def test_determines_reports_conflicting_groups() -> None:
    """Every `a` value must map to a single `b` value."""
    schema = fl.Schema(
        {"a": fl.Column(pl.String), "b": fl.Column(pl.Int64)},
        dataframe_checks=[fl.determines("a", "b")],
    )
    report = schema.validate(
        pl.DataFrame({"a": ["x", "x", "y"], "b": [1, 2, 3]}), raise_on_error=False
    )
    assert not report.is_valid()
    assert report.errors[0].rows == [0, 1]
    assert schema.validate(pl.DataFrame({"a": ["x", "x", "y"], "b": [1, 1, 2]})) is None


def test_within_temporal_gap() -> None:
    """Bound the gap between two temporal columns with a timedelta."""
    schema = fl.Schema(
        {
            "start": fl.Column(pl.Date),
            "end": fl.Column(pl.Date),
        },
        dataframe_checks=[fl.within("start", "end", timedelta(days=3))],
    )
    epoch = date(1970, 1, 1)
    assert (
        schema.validate(pl.DataFrame({"start": [epoch], "end": [epoch + timedelta(days=3)]}))
        is None
    )
    report = schema.validate(
        pl.DataFrame({"start": [epoch], "end": [epoch + timedelta(days=4)]}), raise_on_error=False
    )
    assert not report.is_valid()
    assert report.errors[0].rows == [0]


def test_new_table_checks_json_round_trip() -> None:
    """All five cross-column checks survive to_json/from_json."""
    schema = fl.Schema(
        {
            "a": fl.Column(pl.Int64),
            "b": fl.Column(pl.Int64),
            "start": fl.Column(pl.Date),
            "end": fl.Column(pl.Date),
            "free": fl.Column(pl.String),
        },
        dataframe_checks=[
            fl.ordered(["a", "b"], strict=False),
            fl.sum_columns(["a", "a"], "b"),
            fl.implies("a", "free"),
            fl.determines("a", "free"),
            fl.within("start", "end", timedelta(days=2)),
        ],
    )
    restored = fl.Schema.from_json(schema.to_json())
    assert (
        restored.validate(
            pl.DataFrame(
                {
                    "a": [1, 2],
                    "b": [2, 4],
                    "start": [date(1970, 1, 1), date(1970, 1, 1)],
                    "end": [date(1970, 1, 3), date(1970, 1, 2)],
                    "free": ["x", "x"],
                }
            )
        )
        is None
    )
    report = restored.validate(
        pl.DataFrame(
            {
                "a": [5, 1],
                "b": [2, 4],
                "start": [date(1970, 1, 1), date(1970, 1, 1)],
                "end": [date(1970, 1, 9), date(1970, 1, 2)],
                "free": ["x", "x"],
            }
        ),
        raise_on_error=False,
    )
    assert not report.is_valid()


def test_new_table_checks_generation() -> None:
    """sample() produces frames satisfying the new cross-column checks."""
    schema = fl.Schema(
        {
            "a": fl.Column(pl.Int64, checks=[fl.gt(0)]),
            "b": fl.Column(pl.Int64),
            "start": fl.Column(pl.Date),
            "end": fl.Column(pl.Date),
            "left": fl.Column(pl.Int64, nullable=False),
            "by": fl.Column(pl.String),
            "dup": fl.Column(pl.String),
            "code": fl.Column(pl.Int64),
        },
        dataframe_checks=[
            fl.ordered(["a", "b"], strict=False),
            fl.sum_columns(["a", "a"], "b"),
            fl.implies("left", "by"),
            fl.determines("dup", "code"),
            fl.within("start", "end", timedelta(days=30)),
        ],
    )
    data = fl.sample(schema, 25, seed=7)
    assert schema.validate(data.cast({"start": pl.Date, "end": pl.Date})) is None


def test_nested_list_checks_and_json() -> None:
    """Validate list lengths and persist nested dtypes."""
    schema = fl.Schema(
        {
            "tags": fl.Column(
                pl.List(pl.Int64),
                checks=[fl.list_len_between(1, 2), fl.list_not_empty()],
            )
        }
    )
    valid = pl.DataFrame({"tags": [[1, 2], [3]]})
    assert schema.validate(valid) is None

    invalid = pl.DataFrame({"tags": [[1, 2, 3], []]})
    report = schema.validate(invalid, raise_on_error=False)
    assert any(error.column == "tags" for error in report.errors)

    restored = fl.Schema.from_json(schema.to_json())
    assert restored.validate(valid) is None


def test_struct_field_checks() -> None:
    """Validate struct subfields with regular checks."""
    schema = fl.Schema(
        {
            "address": fl.Column(
                pl.Struct({"city": pl.String, "zip": pl.String}),
                checks=[fl.struct_field("zip", [fl.matches(r"^\d{5}$")])],
            )
        }
    )
    valid = pl.DataFrame({"address": [{"city": "Paris", "zip": "75001"}]})
    assert schema.validate(valid) is None

    invalid = pl.DataFrame({"address": [{"city": "Paris", "zip": "bad"}]})
    report = schema.validate(invalid, raise_on_error=False)
    assert not report.is_valid()
    assert report.errors[0].rows == [0]


def test_infer_schema_list_lengths() -> None:
    """Infer starting list-length bounds from observed data."""
    dataframe = pl.DataFrame({"tags": [[1, 2], [3]]})
    schema = fl.infer_schema(dataframe)
    assert schema.validate(dataframe) is None
    report = schema.validate(pl.DataFrame({"tags": [[1, 2, 3, 4, 5]]}), raise_on_error=False)
    assert any(error.column == "tags" for error in report.errors)


def test_lazy_flag_changes_validate_default() -> None:
    """A lazy schema returns reports instead of raising by default."""
    schema = fl.Schema(
        {"value": fl.Column(pl.Int64, checks=[fl.gt(0)])},
        lazy=True,
    )
    assert schema.lazy is True
    invalid = pl.DataFrame({"value": [-1]})

    report = schema.validate(invalid)
    assert report is not None
    assert not report.is_valid()

    with pytest.raises(ValueError, match="Validation failed"):
        schema.validate(invalid, raise_on_error=True)

    eager = fl.Schema({"value": fl.Column(pl.Int64, checks=[fl.gt(0)])})
    assert eager.lazy is False
    with pytest.raises(ValueError, match="Validation failed"):
        eager.validate(invalid)


def test_validate_lazy_never_raises() -> None:
    """validate_lazy always returns a report object."""
    schema = fl.Schema({"value": fl.Column(pl.Int64, checks=[fl.gt(0)])})
    assert schema.validate_lazy(pl.DataFrame({"value": [1]})) is None
    report = schema.validate_lazy(pl.DataFrame({"value": [-1]}))
    assert report is not None
    assert not report.is_valid()


def test_chunked_validation_reports_absolute_rows_and_never_raises() -> None:
    """chunk_rows validates in batches with absolute row indices."""
    dataframe = pl.DataFrame({"value": [1, 2, -1, 4, 5, 6, -2, 8]})  # rows 2, 6 fail
    schema = fl.Schema({"value": fl.Column(pl.Int64, checks=[fl.gt(0)])})

    # Default mode still raises.
    with pytest.raises(ValueError):
        schema.validate(dataframe)

    # Chunked mode always reports, never raises.
    report = schema.validate(dataframe, chunk_rows=3)
    assert report is not None
    assert not report.is_valid()
    assert sorted(row for error in report.errors for row in error.rows) == [2, 6]

    # Works for LazyFrame inputs too.
    lazy_report = schema.validate(dataframe.lazy(), chunk_rows=3)
    assert lazy_report is not None
    assert sorted(row for error in lazy_report.errors for row in error.rows) == [2, 6]

    assert schema.validate(pl.DataFrame({"value": [1, 2, 3]}), chunk_rows=2) is None


def test_report_raise_if_invalid() -> None:
    """Reports can re-raise on demand."""
    schema = fl.Schema({"value": fl.Column(pl.Int64, checks=[fl.gt(0)])})
    valid_report = schema.validate_lazy(pl.DataFrame({"value": [1]}))
    assert valid_report is None
    report = schema.validate(pl.DataFrame({"value": [-1]}), raise_on_error=False)
    with pytest.raises(ValueError, match="Validation failed"):
        report.raise_if_invalid()


def test_lazy_json_round_trip() -> None:
    """The lazy flag survives JSON persistence."""
    schema = fl.Schema({"value": fl.Column(pl.Int64, checks=[fl.gt(0)])}, lazy=True)
    restored = fl.Schema.from_json(schema.to_json())
    assert restored.lazy is True
    assert restored.validate(pl.DataFrame({"value": [-1]})) is not None


def test_sample_generates_valid_data() -> None:
    """Sampled frames satisfy scalar, table, and nested checks."""
    schema = fl.Schema(
        {
            "age": fl.Column(pl.Int64, nullable=False, checks=[fl.gt(0), fl.lt(120)]),
            "country": fl.Column(pl.String, checks=[fl.one_of(["FR", "US"])]),
            "start": fl.Column(pl.Int64),
            "end": fl.Column(pl.Int64),
            "tags": fl.Column(pl.List(pl.Int64), checks=[fl.list_len_between(1, 2)]),
        },
        dataframe_checks=[fl.lt_column("start", "end")],
    )
    frame = schema.sample(20, seed=42)
    assert frame.height == 20
    assert schema.validate(frame) is None

    again = schema.sample(20, seed=42)
    assert frame.equals(again)

    top_level = fl.sample(schema, 5, seed=1)
    assert top_level.height == 5
    assert schema.validate(top_level) is None


def test_sample_unsatisfiable_raises() -> None:
    """Contradictory bounds surface as Python errors."""
    schema = fl.Schema(
        {"value": fl.Column(pl.Int64, checks=[fl.gt(10), fl.lt(0)])},
    )
    with pytest.raises(ValueError, match="[Uu]nsatisfiable"):
        schema.sample(5, seed=1)


def test_wide_integers_compare_exactly() -> None:
    """Values above 2**53 keep their distinctness in numeric checks."""
    two_to_53 = 2**53
    schema = fl.Schema({"value": fl.Column(pl.UInt64, checks=[fl.gt(float(two_to_53))])})
    dataframe = pl.DataFrame({"value": [two_to_53, two_to_53 + 1]}, schema={"value": pl.UInt64})
    report = schema.validate(dataframe, raise_on_error=False)
    assert not report.is_valid()
    assert report.errors[0].rows == [0]


def test_explicit_not_null_fails_on_all_null_column() -> None:
    """An explicit not_null() still fires on an all-null column."""
    schema = fl.Schema({"value": fl.Column(pl.Null, checks=[fl.not_null()])})
    dataframe = pl.DataFrame({"value": [None, None]}, schema={"value": pl.Null})
    report = schema.validate(dataframe, raise_on_error=False)
    assert not report.is_valid()
    assert report.errors[0].check == "NotNull"
    assert report.errors[0].rows == [0, 1]


def test_pydantic_string_length_is_not_a_list_check() -> None:
    """min_length on str fields must not produce ListLen checks."""

    class Named(BaseModel):
        name: str = Field(min_length=2)

    schema = fl.Schema.from_pydantic(Named)
    assert schema.validate(pl.DataFrame({"name": ["a", "ab"]})) is None


def test_nested_dtype_json_round_trip() -> None:
    """Lists of structs and struct fields survive JSON persistence."""
    schema = fl.Schema(
        {
            "items": fl.Column(pl.List(pl.Struct({"n": pl.Int64}))),
            "address": fl.Column(
                pl.Struct({"zip": pl.String, "codes": pl.List(pl.Int64)}),
                checks=[fl.struct_field("zip", [fl.matches(r"^\d{5}$")])],
            ),
            "nested": fl.Column(pl.Struct({"outer": pl.Struct({"a": pl.Int64})})),
            "deep": fl.Column(pl.Struct({"outer": pl.Struct({"a": pl.Int64, "b": pl.String})})),
            "deeper": fl.Column(pl.Struct({"l1": pl.Struct({"l2": pl.Struct({"x": pl.Int64})})})),
        }
    )
    valid = pl.DataFrame(
        {
            "items": [[{"n": 1}], [{"n": 2}]],
            "address": [
                {"zip": "75001", "codes": [1]},
                {"zip": "69001", "codes": [2, 3]},
            ],
            "nested": [{"outer": {"a": 1}}, {"outer": {"a": 2}}],
            "deep": [
                {"outer": {"a": 1, "b": "x"}},
                {"outer": {"a": 2, "b": "y"}},
            ],
            "deeper": [{"l1": {"l2": {"x": 1}}}, {"l1": {"l2": {"x": 2}}}],
        }
    )
    assert schema.validate(valid) is None
    restored = fl.Schema.from_json(schema.to_json())
    assert restored.validate(valid) is None


def test_struct_names_with_separators_round_trip() -> None:
    """Field names containing commas survive JSON persistence."""
    schema = fl.Schema({"weird": fl.Column(pl.Struct({"a,b": pl.Int64}))})
    valid = pl.DataFrame({"weird": [{"a,b": 1}]})
    assert schema.validate(valid) is None
    restored = fl.Schema.from_json(schema.to_json())
    assert restored.validate(valid) is None


def test_deep_nesting_round_trip_and_sample() -> None:
    """Deeply nested dtypes persist through JSON and sample() builds them."""
    schema = fl.Schema(
        {
            "matrix": fl.Column(pl.List(pl.List(pl.Int64))),
            "grid": fl.Column(pl.List(pl.Struct({"a": pl.Int64, "b": pl.String}))),
            "addresses": fl.Column(
                pl.Struct({"zip": pl.String, "codes": pl.List(pl.List(pl.Int64))}),
                checks=[fl.struct_field("zip", [fl.matches(r"^\d{5}$")])],
            ),
        }
    )
    restored = fl.Schema.from_json(schema.to_json())
    data = fl.sample(restored, 5, seed=5)
    assert restored.validate(data) is None


def test_struct_field_error_lists_full_field_path() -> None:
    """Nested struct field checks compose the complete field path."""
    schema = fl.Schema(
        {
            "l1": fl.Column(
                pl.Struct(
                    {"l2": pl.Struct({"x": pl.Int64, "inner": pl.Struct({"leaf": pl.String})})}
                ),
                checks=[
                    fl.struct_field(
                        "l2",
                        [fl.struct_field("inner", [fl.struct_field("leaf", [fl.not_blank()])])],
                    )
                ],
            )
        }
    )
    dataframe = pl.DataFrame(
        {
            "l1": [
                {"l2": {"x": 1, "inner": {"leaf": "ok"}}},
                {"l2": {"x": 2, "inner": {"leaf": "   "}}},
            ]
        }
    )
    report = schema.validate(dataframe, raise_on_error=False)
    blank_errors = [error for error in report.errors if error.check == "NotBlank"]
    assert len(blank_errors) == 1
    assert blank_errors[0].column == "l1"
    assert blank_errors[0].rows == [1]
    assert "Field 'l2'" in blank_errors[0].message
    assert "Field 'inner'" in blank_errors[0].message


def test_result_classes_are_importable() -> None:
    """ValidationError/Report are part of the top-level API."""
    from framelaw import ValidationError, ValidationReport

    assert "ValidationError" in fl.__all__
    assert "ValidationReport" in fl.__all__
    assert ValidationError is not None and ValidationReport is not None

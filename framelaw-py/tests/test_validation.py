from datetime import date, datetime
from typing import Literal

import framelaw as fl
import polars as pl
import pytest
from pydantic import BaseModel, Field


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
        }
    )
    valid = pl.DataFrame(
        {
            "items": [[{"n": 1}], [{"n": 2}]],
            "address": [
                {"zip": "75001", "codes": [1]},
                {"zip": "69001", "codes": [2, 3]},
            ],
        }
    )
    assert schema.validate(valid) is None
    restored = fl.Schema.from_json(schema.to_json())
    assert restored.validate(valid) is None

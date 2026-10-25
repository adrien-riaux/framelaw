"""Core Framelaw workflows end to end: schemas, inference, and reports."""

from datetime import date
from typing import Literal

import polars as pl
from framelaw import Column, Schema, infer_schema
from framelaw.checks.column import gt, lt, not_null
from pydantic import BaseModel, Field


def main() -> None:
    """Exercise the main framelaw v0.1 workflows."""
    # Explicit schemas support native checks and ordinary DataFrames.
    schema = Schema(
        {
            "age": Column(dtype=pl.Float64, nullable=False, checks=[gt(0.0)]),
            "name": Column(dtype=pl.String, nullable=False, checks=[not_null()]),
        }
    )

    valid_df = pl.DataFrame(
        {"age": [10.0, 20.0, 30.0], "name": ["Alice", "Bob", "Dave"]},
        schema={"age": pl.Float64, "name": pl.String},
    )
    assert schema.validate(valid_df) is None

    # Inference creates a starting schema that can be refined and persisted.
    inferred = infer_schema(valid_df)
    inferred["age"].add_check(lt(100.0))
    assert inferred.validate(valid_df) is None
    persisted = Schema.from_json(inferred.to_json())
    assert persisted.validate(valid_df) is None

    # LazyFrames are collected at the validation boundary.
    polars_schema = Schema.from_polars_schema({"age": pl.Float64, "name": pl.String})
    assert polars_schema.validate(valid_df.lazy()) is None

    # Pydantic v2 constraints become native framelaw checks.
    class User(BaseModel):
        """Example user model for the validation demo."""

        age: int = Field(gt=0, lt=120)
        email: str = Field(pattern=r".+@.+")
        country: Literal["FR", "US"] | None = None

    pydantic_schema = Schema.from_pydantic(User)
    pydantic_df = pl.DataFrame(
        {
            "age": [30],
            "email": ["alice@example.com"],
            "country": ["FR"],
        }
    )
    assert pydantic_schema.validate(pydantic_df) is None

    # Temporal dtypes and their JSON representation are supported too.
    temporal_schema = Schema({"day": Column(pl.Date, checks=[gt(0)])})
    temporal_df = pl.DataFrame({"day": [date(1970, 1, 2), date(1970, 1, 3)]})
    restored_temporal = Schema.from_json(temporal_schema.to_json())
    assert restored_temporal.validate(temporal_df) is None

    invalid_df = pl.DataFrame(
        {"age": [10.0, 20.0, -5.0, 30.0], "name": ["Alice", "Bob", None, "Dave"]}
    )
    report = schema.validate(invalid_df, raise_on_error=False)
    assert report is not None
    assert not report.is_valid()
    assert any(error.column == "age" and error.rows == [2] for error in report.errors)
    assert any(error.column == "name" and error.rows == [2] for error in report.errors)
    report_df = report.to_dataframe()
    assert report_df.columns == ["column", "columns", "check", "row", "message"]
    assert report_df.height == 2
    print(report.summary())

    try:
        schema.validate(invalid_df)
    except ValueError as error:
        print("\nValidation raised as expected:")
        print(error)
    else:
        raise AssertionError("invalid data should raise by default")


if __name__ == "__main__":
    main()

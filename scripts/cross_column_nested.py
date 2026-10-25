"""Framelaw example: cross-column checks and nested types."""

import polars as pl
from framelaw import Column, Schema
from framelaw.checks.column import (
    list_len_between,
    list_not_empty,
    matches,
    struct_field,
)
from framelaw.checks.table import lt_column, unique_together
from pydantic import BaseModel, Field


def main() -> None:
    """Exercise cross-column comparisons, unique-together, and nested checks."""
    schema = Schema(
        {
            "start": Column(pl.Int64),
            "end": Column(pl.Int64),
            "tags": Column(
                pl.List(pl.Int64),
                checks=[list_len_between(1, 3), list_not_empty()],
            ),
            "address": Column(
                pl.Struct({"city": pl.String, "zip": pl.String}),
            ),
        },
        dataframe_checks=[
            lt_column("start", "end"),
            unique_together(["start", "end"]),
        ],
    )

    valid = pl.DataFrame(
        {
            "start": [1, 2, 3],
            "end": [2, 3, 4],
            "tags": [[1], [1, 2], [3]],
            "address": [
                {"city": "Paris", "zip": "75001"},
                {"city": "Lyon", "zip": "69001"},
                {"city": "Nice", "zip": "06000"},
            ],
        }
    )
    assert schema.validate(valid) is None
    print("valid frame passed")

    invalid = pl.DataFrame(
        {
            "start": [5, 2, 2],
            "end": [2, 3, 3],
            "tags": [[1, 2, 3, 4], [], [1]],
            "address": [
                {"city": "Paris", "zip": "75001"},
                {"city": "Lyon", "zip": "69001"},
                {"city": "Nice", "zip": "06000"},
            ],
        }
    )
    report = schema.validate(invalid, raise_on_error=False)
    assert report is not None
    assert not report.is_valid()
    print(report.summary())
    print(report.to_dataframe())

    # Zip-code pattern on a struct subfield.
    zip_schema = Schema(
        {
            "address": Column(
                pl.Struct({"city": pl.String, "zip": pl.String}),
                checks=[struct_field("zip", [matches(r"^\d{5}$")])],
            )
        }
    )
    assert zip_schema.validate(valid) is None
    bad_zip = pl.DataFrame({"address": [{"city": "Paris", "zip": "bad"}]})
    bad_report = zip_schema.validate(bad_zip, raise_on_error=False)
    assert bad_report is not None
    assert not bad_report.is_valid()
    print("struct field check:", bad_report.summary())

    # JSON round-trip preserves table checks and nested dtypes/checks.
    restored = Schema.from_json(schema.to_json())
    assert restored.validate(valid) is None
    restored_report = restored.validate(invalid, raise_on_error=False)
    assert restored_report is not None and not restored_report.is_valid()
    print("JSON round-trip passed")

    # Pydantic list fields map to List dtypes.
    class WithTags(BaseModel):
        """Tagged-values model for the Pydantic list demo."""

        values: list[int] = Field(min_length=1, max_length=3)

    pydantic_schema = Schema.from_pydantic(WithTags)
    assert pydantic_schema.validate(pl.DataFrame({"values": [[1, 2]]})) is None
    print("pydantic list passed")


if __name__ == "__main__":
    main()

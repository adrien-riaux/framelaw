"""Inspect validation failures as structured data."""

import polars as pl
from framelaw import Column, Schema
from framelaw.checks.column import gt, one_of


def main() -> None:
    """Show every way to read a validation report."""
    schema = Schema(
        {
            "age": Column(pl.Int64, nullable=False, checks=[gt(0)]),
            "country": Column(pl.String, checks=[one_of(["FR", "US"])]),
        }
    )
    df = pl.DataFrame({"age": [28, -4], "country": ["FR", "XX"]})

    report = schema.validate(df, raise_on_error=False)
    assert report is not None
    print("valid:", report.is_valid())
    print(report.summary())

    for error in report.errors:
        print(f"{error.columns} | {error.check} | rows={error.rows}")
        print(f"  {error.message}")

    failures = report.to_dataframe()
    print(failures)
    assert failures.columns == ["column", "columns", "check", "row", "message"]

    try:
        report.raise_if_invalid()
    except ValueError as err:
        print("raised as expected:")
        print(err)
    else:
        raise AssertionError("report should have raised")


if __name__ == "__main__":
    main()

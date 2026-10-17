"""Framelaw v0.3 example: lazy validation mode and data generation."""

import polars as pl
from framelaw import Column, Schema, gt, list_len_between, lt_column, one_of, sample


def main() -> None:
    """Exercise lazy defaults, validate_lazy, and schema sampling."""
    schema = Schema(
        {
            "age": Column(pl.Int64, nullable=False, checks=[gt(0)]),
            "country": Column(pl.String, checks=[one_of(["FR", "US"])]),
            "start": Column(pl.Int64),
            "end": Column(pl.Int64),
            "tags": Column(pl.List(pl.Int64), checks=[list_len_between(1, 2)]),
        },
        dataframe_checks=[lt_column("start", "end")],
        lazy=True,
    )

    # Lazy schemas return reports instead of raising by default.
    bad = pl.DataFrame(
        {
            "age": [-1, 30],
            "country": ["FR", "DE"],
            "start": [1, 5],
            "end": [2, 3],
            "tags": [[1], [1, 2, 3]],
        }
    )
    report = schema.validate(bad)
    assert report is not None and not report.is_valid()
    print(report.summary())

    try:
        report.raise_if_invalid()
    except ValueError as error:
        print("\nraise_if_invalid raised as expected:")
        print(error)
    else:
        raise AssertionError("report should have raised")

    # validate_lazy never raises, even on eager schemas.
    eager = Schema({"age": Column(pl.Int64, checks=[gt(0)])})
    assert eager.validate_lazy(bad.select("age")) is not None

    # Generate valid data from the schema and validate it.
    frame = schema.sample(10, seed=42)
    assert schema.validate(frame, raise_on_error=True) is None
    print("\nsampled frame:")
    print(frame)

    again = schema.sample(10, seed=42)
    assert frame.equals(again)
    print("\nseeded sampling is deterministic")

    top_level = sample(schema, 5, seed=1)
    assert top_level.height == 5
    assert schema.validate(top_level) is None
    print("top-level sample() works")

    # Lazy flag survives JSON persistence.
    restored = Schema.from_json(schema.to_json())
    assert restored.lazy is True
    assert restored.validate(bad) is not None
    print("lazy JSON round-trip passed")


if __name__ == "__main__":
    main()

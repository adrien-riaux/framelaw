"""Generate valid test data from a schema."""

import polars as pl
from framelaw import Column, Schema, gt, lt, lt_column, one_of, sample


def main() -> None:
    """Sample reproducible fixtures and self-validate them."""
    schema = Schema(
        {
            "age": Column(pl.Int64, nullable=False, checks=[gt(0), lt(120)]),
            "country": Column(pl.String, checks=[one_of(["FR", "US"])]),
            "start": Column(pl.Int64),
            "end": Column(pl.Int64),
        },
        dataframe_checks=[lt_column("start", "end")],
    )

    frame = schema.sample(10, seed=42)
    assert schema.validate(frame) is None
    print(frame)

    assert frame.equals(schema.sample(10, seed=42))
    print("same seed, same frame")

    other = sample(schema, 5, seed=7)
    assert isinstance(other, pl.DataFrame) and other.height == 5
    assert schema.validate(other) is None
    print("top-level sample() works")


if __name__ == "__main__":
    main()

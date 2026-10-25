"""Benchmark validation throughput (use a release build for real numbers)."""

import time

import polars as pl
from framelaw import Column, Schema
from framelaw.checks.column import gt, lt, matches, one_of, unique


def bench(label: str, schema: Schema, df: pl.DataFrame, rounds: int = 5) -> None:
    """Time best-of-N validation runs and print throughput."""
    best = float("inf")
    for _ in range(rounds):
        started = time.perf_counter()
        assert schema.validate(df) is None
        best = min(best, time.perf_counter() - started)
    print(f"{label}: {df.height / best:,.0f} rows/s ({best:.3f}s best of {rounds})")


def main() -> None:
    """Validate a 2M-row frame with mixed checks and varied strings."""
    n = 2_000_000
    df = pl.DataFrame(
        {
            "id": range(n),
            "age": [(i % 100) + 1 for i in range(n)],
            "email": [f"user{i}@example.com" for i in range(n)],
            "country": (["FR", "US", "DE"] * ((n // 3) + 1))[:n],
        },
        schema={
            "id": pl.Int64,
            "age": pl.Int64,
            "email": pl.String,
            "country": pl.String,
        },
    )
    schema = Schema(
        {
            "id": Column(pl.Int64, checks=[unique()]),
            "age": Column(pl.Int64, checks=[gt(0), lt(120)]),
            "email": Column(pl.String, checks=[matches(r".+@.+\..+")]),
            "country": Column(pl.String, checks=[one_of(["FR", "US", "DE"])]),
        }
    )
    bench("mixed checks, 2M rows", schema, df)


if __name__ == "__main__":
    main()

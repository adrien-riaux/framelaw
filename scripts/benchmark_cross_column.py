"""Benchmark the cross-column table checks (use a release build for numbers)."""

import time
from datetime import timedelta

import polars as pl
from framelaw import Column, Schema
from framelaw.checks.table import determines, implies, ordered, sum_columns, within


def bench(label: str, schema: Schema, df: pl.DataFrame, rounds: int = 5) -> None:
    """Time best-of-N validation runs and print throughput."""
    best = float("inf")
    for _ in range(rounds):
        started = time.perf_counter()
        assert schema.validate(df) is None
        best = min(best, time.perf_counter() - started)
    print(f"{label}: {df.height / best:,.0f} rows/s ({best:.3f}s best of {rounds})")


def main() -> None:
    """Validate a 1M-row frame carrying every cross-column check."""
    n = 1_000_000
    epoch_day = 19000
    df = pl.DataFrame(
        {
            "a": [i % 1000 for i in range(n)],
            "b": [i % 1000 for i in range(n)],
            "left": [i % 100 if i % 7 else None for i in range(n)],
            "by": [i % 100 for i in range(n)],
            "code": [i % 500 for i in range(n)],
            "start": [epoch_day + (i % 30) for i in range(n)],
            "end": [epoch_day + (i % 30) + 5 for i in range(n)],
        },
        schema={
            "a": pl.Int64,
            "b": pl.Int64,
            "left": pl.Int64,
            "by": pl.Int64,
            "code": pl.Int64,
            "start": pl.Int32,
            "end": pl.Int32,
        },
    ).with_columns(pl.col("start").cast(pl.Date), pl.col("end").cast(pl.Date))

    schema = Schema(
        {
            "a": Column(pl.Int64),
            "b": Column(pl.Int64),
            "left": Column(pl.Int64),
            "by": Column(pl.Int64),
            "code": Column(pl.Int64),
            "start": Column(pl.Date),
            "end": Column(pl.Date),
        },
        dataframe_checks=[
            ordered(["a", "b"], strict=False),
            sum_columns(["a"], "b"),
            implies("left", "by"),
            determines("code", "by"),
            within("start", "end", timedelta(days=30)),
        ],
    )
    bench("cross-column checks, 1M rows", schema, df)


if __name__ == "__main__":
    main()

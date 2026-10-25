"""Benchmark temporal inference, validation, and bounded generation."""

import time
from datetime import date, timedelta

import polars as pl
from framelaw import Column, Schema, infer_schema, sample, within

ROUNDS = 5


def bench_validate(label: str, schema: Schema, df: pl.DataFrame) -> None:
    """Time best-of-N validation runs and print throughput."""
    best = float("inf")
    for _ in range(ROUNDS):
        started = time.perf_counter()
        assert schema.validate(df) is None
        best = min(best, time.perf_counter() - started)
    print(f"{label}: {df.height / best:,.0f} rows/s ({best:.3f}s best of {ROUNDS})")


def main() -> None:
    """Infer tick bounds from a temporal frame and validate/sample against them."""
    n = 1_000_000
    epoch = date(1970, 1, 1)
    df = pl.DataFrame(
        {
            "start": [i % 3650 for i in range(n)],
            "end": [i % 3650 + 10 for i in range(n)],
        },
        schema={"start": pl.Int32, "end": pl.Int32},
    ).select(
        (pl.lit(epoch) + pl.duration(days=pl.col("start"))).alias("start"),
        (pl.lit(epoch) + pl.duration(days=pl.col("end"))).alias("end"),
    )

    schema = infer_schema(df)
    # Inference emits gt/lt tick bounds around the observed date span; the
    # inferred schema must accept the source data and stay stable.
    assert schema.validate(df) is None
    bench_validate("inferred temporal bounds, 1M rows", schema, df)

    started = time.perf_counter()
    generated = sample(schema, n, seed=7)
    elapsed = time.perf_counter() - started
    assert schema.validate(generated) is None
    print(
        f"sample temporal bounds, {n:,} rows: {n / elapsed:,.0f} rows/s ({elapsed:.3f}s)"
    )

    gap_schema = Schema(
        {"start": Column(pl.Date), "end": Column(pl.Date)},
        dataframe_checks=[within("start", "end", timedelta(days=60))],
    )
    bench_validate("within gap check, 1M rows", gap_schema, df)


if __name__ == "__main__":
    main()

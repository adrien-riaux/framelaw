"""Benchmark deeply nested dtype validation and generation."""

import time

import polars as pl
from framelaw import Column, Schema, sample

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
    """Validate and then sample 50k rows of List(List) and List(Struct) data."""
    n = 50_000
    df = pl.DataFrame(
        {
            "matrix": [[[i % 97, (i + 1) % 97], [i % 13]] for i in range(n)],
            "grid": [[{"a": i % 11, "b": f"v{i % 5}"}] * 2 for i in range(n)],
        },
        schema={
            "matrix": pl.List(pl.List(pl.Int64)),
            "grid": pl.List(pl.Struct({"a": pl.Int64, "b": pl.String})),
        },
    )
    schema = Schema(
        {
            "matrix": Column(pl.List(pl.List(pl.Int64))),
            "grid": Column(pl.List(pl.Struct({"a": pl.Int64, "b": pl.String}))),
        }
    )
    bench_validate("nested List(List) + List(Struct), 50k rows", schema, df)

    started = time.perf_counter()
    generated = sample(schema, n, seed=7)
    elapsed = time.perf_counter() - started
    assert schema.validate(generated) is None
    print(
        f"sample nested dtypes, {n:,} rows: {n / elapsed:,.0f} rows/s ({elapsed:.3f}s)"
    )


if __name__ == "__main__":
    main()

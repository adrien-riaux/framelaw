"""Benchmark custom expression, batch-mask, and row checks (use a release build)."""

import time

import polars as pl
from framelaw import Column, Schema
from framelaw.checks.table import check_expr, check_mask, check_rows, sum_columns


def bench(
    label: str,
    schema: Schema,
    frame: pl.DataFrame | pl.LazyFrame,
    rows: int,
    *,
    chunk_rows: int | None = None,
    rounds: int = 5,
) -> None:
    """Time best-of-N validation runs and print throughput."""
    best = float("inf")
    for _ in range(rounds):
        started = time.perf_counter()
        assert schema.validate(frame, chunk_rows=chunk_rows) is None
        best = min(best, time.perf_counter() - started)
    print(f"{label}: {rows / best:,.0f} rows/s ({best:.3f}s best of {rounds})")


def main() -> None:
    """Compare the native check with the three custom-check tiers on 1M rows."""
    n = 1_000_000
    quantities = [(i % 100) + 1 for i in range(n)]
    df = pl.DataFrame(
        {
            "qty": quantities,
            "price": [1.5] * n,
            "total": [quantity + 1.5 for quantity in quantities],
        },
        schema={"qty": pl.Int64, "price": pl.Float64, "total": pl.Float64},
    )
    columns = {
        "qty": Column(pl.Int64),
        "price": Column(pl.Float64),
        "total": Column(pl.Float64),
    }

    native = Schema(columns, dataframe_checks=[sum_columns(["qty", "price"], "total")])
    expression = Schema(
        columns,
        dataframe_checks=[
            check_expr(
                (pl.col("qty") + pl.col("price") - pl.col("total")).abs() < 1e-9,
                name="total_consistent",
            )
        ],
    )
    batch = Schema(
        columns,
        dataframe_checks=[
            check_mask(
                lambda frame: (
                    (frame["qty"] + frame["price"] - frame["total"]).abs() < 1e-9
                ),
                columns=["qty", "price", "total"],
                name="total_consistent",
            )
        ],
    )
    row = Schema(
        columns,
        dataframe_checks=[
            check_rows(
                lambda record: (
                    abs(record["qty"] + record["price"] - record["total"]) < 1e-9
                ),
                columns=["qty", "price", "total"],
                name="total_consistent",
            )
        ],
    )

    bench("native sum_columns, 1M rows", native, df, n)
    bench("check_expr, 1M rows", expression, df, n)
    bench("check_mask (vectorized), 1M rows", batch, df, n)
    bench("check_rows (per row), 1M rows", row, df, n)
    bench(
        "check_mask, lazy streaming (100k chunks)",
        batch,
        df.lazy(),
        n,
        chunk_rows=100_000,
    )
    bench(
        "check_rows, lazy streaming (100k chunks)",
        row,
        df.lazy(),
        n,
        chunk_rows=100_000,
    )


if __name__ == "__main__":
    main()

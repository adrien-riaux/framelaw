# Custom checks

The built-in checks cover common contracts, but domain rules often need
something more specific. Three constructors let you express custom rules
without leaving Polars:

| Constructor | Input | Runs |
|---|---|---|
| `fl.checks.table.check_expr(expr)` | `polars.Expr` (or a zero-argument callable) | Vectorized Polars expression |
| `fl.checks.table.check_mask(func)` | `func(polars.DataFrame) -> mask` | Once per validation chunk |
| `fl.checks.table.check_rows(func)` | `func(dict) -> bool` | Once per row |

All three return a `DataFrameCheck` and attach through
`Schema(..., dataframe_checks=[...])`.

## Expression checks

An expression check is the fastest option: it is evaluated by Polars itself,
uses every core, and allocates only one Boolean value per row.

```python
import polars as pl
import framelaw as fl

schema = fl.Schema(
    {
        "qty": fl.Column(pl.Int64),
        "price": fl.Column(pl.Float64),
        "total": fl.Column(pl.Float64),
    },
    dataframe_checks=[
        fl.checks.table.check_expr(
            (pl.col("qty") * pl.col("price") - pl.col("total")).abs() <= 0.01,
            name="total_consistent",
        ),
    ],
)

df = pl.DataFrame(
    {
        "qty": [2, 3],
        "price": [10.0, 4.0],
        "total": [20.0, 12.0],
    }
)
assert schema.validate(df) is None
```

The expression must produce exactly one Boolean column with the same height as
the validated frame. `False` fails the row and nulls are skipped. Aggregating
expressions are rejected because they do not yield a per-row mask.

## Batch mask checks

Use `check_mask` when the rule needs Python: external lookups, a library that
has no Polars expression equivalent, or state prepared before validation.

```python
import polars as pl
import framelaw as fl

allowed_regions = {"EU", "US"}

schema = fl.Schema(
    {
        "qty": fl.Column(pl.Int64),
        "price": fl.Column(pl.Float64),
        "region": fl.Column(pl.String),
    },
    dataframe_checks=[
        fl.checks.table.check_mask(
            lambda frame: (frame["qty"] > 0)
            & (frame["price"] > 0)
            & frame["region"].is_in(list(allowed_regions)),
            columns=["qty", "price", "region"],
            name="order_line_sane",
        ),
    ],
)

df = pl.DataFrame(
    {
        "qty": [1, 0],
        "price": [9.5, 3.0],
        "region": ["EU", "US"],
    }
)

report = schema.validate(df, raise_on_error=False)
assert report is not None
assert report.errors[0].rows == [1]
```

The callable receives one `polars.DataFrame` per **chunk** — not one row at a
time — and must return a Boolean `polars.Series` or a sequence of booleans.
Returning a Python list is supported but wraps it in a Polars Series, so
prefer the Series. Exceptions raised by the callable propagate unchanged.

## Row checks

Use `check_rows` when the rule is naturally expressed per row and cannot be
vectorized: a small decision tree, a call into a library that only accepts
scalars, or a quick prototype. The callable receives a `dict` with the
selected columns (every column by default) and returns `True`, `False`, or
`None` to skip the row.

```python
schema = fl.Schema(
    {
        "qty": fl.Column(pl.Int64),
        "price": fl.Column(pl.Float64),
        "total": fl.Column(pl.Float64),
    },
    dataframe_checks=[
        fl.checks.table.check_rows(
            lambda row: abs(row["qty"] * row["price"] - row["total"]) <= 0.01,
            columns=["qty", "price", "total"],
            name="total_consistent",
        ),
    ],
)

df = pl.DataFrame(
    {
        "qty": [2, 3],
        "price": [10.0, 4.0],
        "total": [20.0, 12.0],
    }
)
assert schema.validate(df) is None
```

Polars compiles the predicate to `pl.struct(...).map_elements(...)`, so rows
are processed in batches rather than through a Python loop you write. The cost
is still dominated by one Python call per row, and the GIL serializes those
calls, so `check_rows` uses one core. Prefer `check_expr` whenever the rule can
be written as an expression, then `check_mask` when it needs Python but can
process a whole chunk at once.

## Performance

Measured on a 1M-row frame with a simple predicate (Python 3.14, Polars 1.44;
the callback body dominates these numbers):

| Tier | Constructor | Time (1M rows) | Cores | Memory |
|---|---|---|---|---|
| 1 | `check_expr` | ~0.4 ms | all | one Boolean per row |
| 2 | `check_mask` | callback + one GIL crossing per chunk | callback-dependent | one chunk |
| 3 | `check_rows` | ~0.1–0.2 s (~0.1–0.2 µs/row) | one | one Boolean per row |

Tier 3 is roughly 100–300× slower than a vectorized expression, and the gap
grows with the work done per row (Pydantic validation, I/O, regexes). Treat it
as a correctness-first escape hatch, not a throughput path.

## Failures and row limits

Custom failures appear in the report like any other check: the supplied `name`
is the check name, and `columns` (or the expression's referenced columns) are
attributed to the error.

A broken predicate can fail on every row. To keep reports bounded,
`max_reported_rows` (default `1000`) caps the stored row indices; the message
still reports the true total:

```python
counter_schema = fl.Schema(
    {"value": fl.Column(pl.Int64)},
    dataframe_checks=[
        fl.checks.table.check_expr(
            pl.col("value") > 100, name="large_value", max_reported_rows=10
        ),
    ],
)
```

## Chunked validation

Pass `chunk_rows` to bound peak memory. For a `LazyFrame`, framelaw uses
Polars' streaming `collect_batches`, so the query runs once and each batch is
validated as it arrives; for an eager `DataFrame`, or a Polars version without
`collect_batches`, it slices the frame.

```python
report = schema.validate(df.lazy(), chunk_rows=100_000)
assert report is None
```

`Schema.validate_lazy(..., chunk_rows=...)` accepts the same option and never
raises.

Dataset-wide checks (`unique`, `unique_together`, `determines`, `ordered`)
only see one batch at a time. Validate without chunking when those must span
the full frame.

Expression checks keep native speed. Batch callbacks pay one GIL round-trip
per check per chunk — negligible for batches of thousands of rows. Row checks
pay one Python call per row regardless of chunking; chunking bounds memory,
not CPU.

## Limits

- **Serialization.** `Schema.to_json()` raises `ValueError` when the schema
  contains a custom check, because Python callables cannot be persisted.
  Rebuild custom checks in code when loading a schema from JSON.
- **Data generation.** `Schema.sample()` ignores custom checks; they cannot
  guide the generator.
- **Pydantic.** `Schema.from_pydantic()` derives only native checks.

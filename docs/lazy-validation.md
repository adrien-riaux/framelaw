# Lazy validation

By default `validate()` raises `ValueError` on failure — convenient for
scripts, awkward for pipelines that want to inspect, log, or branch on
failures. Lazy mode returns reports instead. For large frames, `chunk_rows`
validates in batches (see [Chunked validation](#chunked-validation)).

```python
import polars as pl
import framelaw as fl

schema = fl.Schema(
    {"id": fl.Column(pl.Int64, nullable=False)},
)
df = pl.DataFrame({"id": [1, None, 3]})
```

## Per-call lazy validation

```python
report = schema.validate(df, raise_on_error=False)
if report is not None and not report.is_valid():
    print(report.summary())
```

`validate_lazy()` is a shorthand that never raises, regardless of settings:

```python
report = schema.validate_lazy(df)  # None if valid, else ValidationReport
```

## Schema-level lazy mode

Pass `lazy=True` to flip the default for that schema. Explicit
`raise_on_error` still wins per call:

```python
schema = fl.Schema(
    {"id": fl.Column(pl.Int64, nullable=False)},
    lazy=True,
)

schema.validate(df)  # report, never raises
try:
    schema.validate(df, raise_on_error=True)  # raises again for this call
except ValueError as err:
    print(err)
schema.lazy  # True
```

The flag persists through `to_json()` / `from_json()`.

## Re-raising from a report

```python
report = schema.validate_lazy(df)
if report is not None:
    try:
        report.raise_if_invalid()  # ValueError with the summary
    except ValueError as err:
        print(err)
```

This makes the common pipeline shape explicit: validate lazily everywhere,
raise exactly where a failure must stop the run.

## Chunked validation

For large frames, `chunk_rows` validates the data in batches instead of one
monolithic pass:

```python
report = schema.validate(df, chunk_rows=100_000)
```

- Works for both `DataFrame` and `LazyFrame` inputs; lazy frames are
  sliced and collected per batch, so batches materialize lazily.
- Each batch runs on the Rust side with the GIL released, keeping the
  producer thread free.
- Error row indices are absolute positions in the whole frame.
- The returned report never raises — even if the schema would raise by
  default. Inspect the report and raise yourself when needed.

Two caveats:

- Dataset-wide checks (`fl.checks.column.unique()`, `fl.unique_together`,
  `fl.determines`) only see one chunk at a time, so violations that span
  chunk boundaries go unnoticed at these row counts.
- A perfectly aligned `chunk_rows` can reduce per-batch overhead; common
  practice is 100k–1M rows per batch.

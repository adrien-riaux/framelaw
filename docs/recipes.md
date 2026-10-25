# Recipes

Short, runnable patterns for common Framelaw workflows. The examples share
this setup:

```python
import polars as pl
import framelaw as fl

orders = fl.Schema(
    {
        "order_id": fl.Column(pl.Int64, nullable=False, checks=[fl.checks.column.unique()]),
        "amount": fl.Column(pl.Float64, nullable=False, checks=[fl.checks.column.gt(0)]),
        "currency": fl.Column(pl.String, checks=[fl.checks.column.one_of(["EUR", "USD", "GBP"])]),
    }
)

frame = pl.DataFrame(
    {
        "order_id": [1, 2, 3],
        "amount": [19.99, 5.0, 42.5],
        "currency": ["EUR", "USD", "GBP"],
    }
)
```

## Gate data at ingestion

Raise as soon as a frame violates the contract:

```python
orders.validate(frame)  # raises ValueError on the first invalid frame
```

For a pipeline that should keep running, use a report instead:

```python
report = orders.validate(frame, raise_on_error=False)
if report is not None:
    log.error("orders failed validation\n%s", report.summary())
```

## Quarantine failing rows

`ValidationReport.to_dataframe()` gives one row per failing cell. Join those
row indices back onto the original frame:

```python
import polars as pl

report = orders.validate(frame, raise_on_error=False)
if report is not None:
    failing_rows = report.to_dataframe().select(pl.col("row").drop_nulls().unique())
    bad = frame.with_row_index("row").join(failing_rows, on="row", how="semi").drop("row")
    good = frame.with_row_index("row").join(failing_rows, on="row", how="anti").drop("row")

    bad.write_parquet("orders.rejected.parquet")
    good.write_parquet("orders.accepted.parquet")
```

Schema-level errors (missing columns, dtype mismatches) have a null `row`;
check `report.summary()` for those before trying to quarantine rows.

## Keep the schema in version control

Serialize once, commit the JSON, and load it wherever the data is validated:

```python
from pathlib import Path

Path("orders.schema.json").write_text(orders.to_json())

# In another service or a test:
loaded = fl.Schema.from_json(Path("orders.schema.json").read_text())
loaded.validate(frame)
```

The loader accepts schema versions 1 and 2, so older contracts keep working.
See [Serialization](serialization.md).

## Derive the schema from Pydantic

Avoid maintaining the same constraints twice:

```python
from pydantic import BaseModel, Field
import framelaw as fl

class User(BaseModel):
    user_id: int
    age: int = Field(gt=0, lt=120)
    email: str = Field(pattern=r".+@.+\..+")

schema = fl.Schema.from_pydantic(User)
schema.validate(pl.DataFrame({"user_id": [1], "age": [30], "email": ["a@example.com"]}))
```

See [Pydantic integration](pydantic.md) for the full type mapping.

## Generate fixtures from the contract

The same schema can produce deterministic test data:

```python
fixtures = orders.sample(500, seed=42)
assert orders.validate(fixtures) is None
```

`sample` raises `ValueError` for contradictory schemas rather than returning
invalid data. See [Data generation](data-generation.md).

## Validate a large LazyFrame in batches

`chunk_rows` slices and collects one batch at a time, keeping memory bounded:

```python
import polars as pl

lazy = pl.scan_parquet("events/*.parquet")
report = orders.validate(lazy, chunk_rows=250_000)
if report is not None:
    print(report.summary())
```

Chunked validation always returns a report instead of raising. Dataset-wide
checks (`unique`, `unique_together`, `determines`) only see one batch at a
time; see [Lazy validation](lazy-validation.md#chunked-validation) for the
caveats.

## Enforce rules across columns

Use table checks for invariants that need more than one column:

```python
import polars as pl
import framelaw as fl

schema = fl.Schema(
    {
        "start": fl.Column(pl.Date, nullable=False),
        "end": fl.Column(pl.Date, nullable=False),
        "country": fl.Column(pl.String, nullable=False),
        "city": fl.Column(pl.String, nullable=False),
    },
    dataframe_checks=[
        fl.checks.table.lt_column("start", "end"),
        fl.checks.table.unique_together(["country", "city"]),
    ],
)
```

See [Cross-column checks](cross-column-checks.md).

## Migrate from the flat check API

Before 0.3, checks were imported directly from `framelaw`. The old names still
work but emit `DeprecationWarning`:

```python
# Deprecated:
fl.gt(0)

# Preferred:
fl.checks.column.gt(0)
```

Table checks follow the same pattern: `fl.checks.table.lt_column(...)`.

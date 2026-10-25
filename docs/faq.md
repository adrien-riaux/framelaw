# FAQ

```python
import polars as pl
import framelaw as fl

schema = fl.Schema(
    {"age": fl.Column(pl.Int64, checks=[fl.checks.column.gt(0)])}
)
df = pl.DataFrame({"age": [30, -1]})
```

## Why does `validate()` return `None` when the frame is valid?

`validate()` is designed to be a statement, not an assertion. A valid frame
needs no further action, so it returns `None`; an invalid frame either raises
`ValueError` or returns a `ValidationReport`, depending on `raise_on_error`.
Use `report = schema.validate(df, raise_on_error=False)` when you want to
inspect failures without exceptions.

## What is the difference between `nullable=False` and `not_null()`?

They enforce the same rule. `nullable=False` is a column-level property that
produces a `NotNull` error; `fl.checks.column.not_null()` is an explicit check
with the same effect. `from_pydantic` and JSON round-trips materialize the
explicit check. Pick one style and stay consistent.

## Why are nulls skipped by value checks?

Value checks (`gt`, `matches`, `one_of`, …) apply to non-null cells only.
Whether a null is allowed is governed by `nullable` / `not_null`. This keeps
the two concerns separate: a nullable column can still enforce a regex on the
values that are present.

## Why is my `Int32` column rejected by an `Int64` schema?

FrameLaw requires an exact dtype match (structural for nested types). `Int32`
and `Int64` are different storage types, and silently widening them would hide
a real schema drift. Use the dtype you actually store, or cast at the
boundary.

## Why did `unique` miss duplicates when using `chunk_rows`?

Chunked validation runs one batch at a time. Dataset-wide checks — `unique`,
`unique_together`, `determines` — only see the current batch, so a duplicate
that straddles a batch boundary is not detected. Validate the full frame when
those checks matter, or deduplicate before chunking.

## Does FrameLaw validate `LazyFrame`s lazily?

`validate_lazy()` and `validate()` collect a `LazyFrame` once at the
validation boundary. With `chunk_rows`, they slice and collect one batch at a
time, so the full frame is never materialized. The validation itself is eager
per batch.

## Can I use FrameLaw with Pandas or Spark?

No. FrameLaw is built on Polars' native columnar buffers and exposes Polars
dtypes in schemas. Pandera or Great Expectations are better fits for other
engines.

## What does `sample()` do when a schema is contradictory?

It raises `ValueError` with a message describing the unsatisfiable
constraint — for example `gt(10)` combined with `lt(0)`, an impossible
`unique` domain, or a regex with no known valid canned value. It never
returns data that would fail its own schema.

## Why does `from_pydantic` ignore `min_length` / `max_length` on strings?

Pydantic string-length constraints are not mapped yet. FrameLaw does have a
string-length check, so add it explicitly after deriving the schema:

```python
from pydantic import BaseModel, Field

class User(BaseModel):
    email: str = Field(pattern=r".+@.+\..+")

schema = fl.Schema.from_pydantic(User)
schema["email"].add_check(fl.checks.column.str_length_between(3, 254))
```

Length constraints on `list` fields are mapped to `list_len_between`.

## Which Pydantic versions are supported?

Pydantic v2 only. FrameLaw reads `__pydantic_core_schema__` directly; v1
models do not expose that structure.

## Why do I get a `DeprecationWarning` for `fl.gt(...)`?

The flat top-level check names were deprecated in 0.3. Use the scoped
namespaces:

```python
fl.checks.column.gt(0)                 # single-column checks
fl.checks.table.lt_column("start", "end")  # cross-column checks
```

`infer_schema` and `sample` remain top-level.

## Can I persist a schema and load it in another language?

The JSON format is documented by the `to_json()` output and versioned. The
current writer emits version 2 and the loader accepts versions 1 and 2. The
format is stable within a major version; see [Serialization](serialization.md).

## How do I get the failing rows as a frame?

```python
report = schema.validate(df, raise_on_error=False)
if report is not None:
    failures = report.to_dataframe()
```

`failures` has columns `column`, `columns`, `check`, `row`, and `message`.
Schema-level errors have a null `row`. See
[Error reports](error-reports.md).

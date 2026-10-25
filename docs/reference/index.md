# API overview

FrameLaw exposes a small surface: schema objects at the top level, check
constructors grouped by scope, and a report type for structured failures.

```python
import polars as pl
import framelaw as fl

df = pl.DataFrame({"age": [30]})
schema = fl.Schema({"age": fl.Column(pl.Int64)})

fl.infer_schema(df)                        # draft a schema
fl.checks.column.gt(0)                     # single-column check
fl.checks.table.lt_column("start", "end")  # cross-column check
fl.sample(schema, 100, seed=42)            # valid fixture rows
```

## Module layout

| Import | Contains |
|---|---|
| `framelaw` | `Schema`, `Column`, `ColumnRef`, `Check`, `DataFrameCheck`, `ValidationReport`, `ValidationError`, `infer_schema`, `sample`, `__version__` |
| `framelaw.checks.column` | Single-column check constructors (`gt`, `matches`, `one_of`, `list_len_between`, …) |
| `framelaw.checks.table` | Cross-column check constructors (`lt_column`, `unique_together`, `ordered`, …) |

## Classes

| Class | Purpose | Details |
|---|---|---|
| [`Schema`](schema.md#framelaw.Schema) | Column specs plus table checks; validate, serialize, sample | [Schema & columns](schema.md) |
| [`Column`](schema.md#framelaw.Column) | One column's dtype, nullability, and checks | [Schema & columns](schema.md) |
| [`ColumnRef`](schema.md#framelaw.ColumnRef) | Mutable reference returned by `schema[name]` | [Schema & columns](schema.md) |
| [`Check`](checks.md#framelaw.Check) | Opaque single-column check value | [Single-column checks](checks.md) |
| [`DataFrameCheck`](table-checks.md#framelaw.DataFrameCheck) | Opaque table-check value | [Table checks](table-checks.md) |
| [`ValidationReport`](reports.md#framelaw.ValidationReport) | Structured validation result | [Reports](reports.md) |
| [`ValidationError`](reports.md#framelaw.ValidationError) | One failure with columns, check, rows, message | [Reports](reports.md) |

## Single-column constructors

All of these live in `framelaw.checks.column` and return a `Check`.

| Function | Summary |
|---|---|
| `not_null()` | Reject null values |
| `unique_values()` / `unique()` | Reject duplicated values |
| `gt(value)` / `lt(value)` / `gte(value)` / `lte(value)` | Numeric or temporal bounds |
| `between(min, max)` | Inclusive numeric or temporal range |
| `matches(pattern)` / `matches_regex(pattern)` | Regex match on non-null strings |
| `one_of(values)` / `is_in(values)` | String, numeric, or boolean membership |
| `str_length_between(min_len, max_len)` | Character-length bounds |
| `not_blank()` | Reject empty and whitespace-only strings |
| `starts_with(prefix)` / `ends_with(suffix)` | String affixes |
| `list_len_between(min_len, max_len)` | `List`/`Array` length bounds |
| `list_not_empty()` | Reject empty `List`/`Array` cells |
| `struct_field(field, checks)` | Apply checks to a `Struct` subfield |

See [Single-column checks](checks.md) for full signatures and semantics.

## Table-check constructors

All of these live in `framelaw.checks.table` and return a `DataFrameCheck`.

| Function | Summary |
|---|---|
| `gt_column(a, b)` / `lt_column(a, b)` / `gte_column(a, b)` / `lte_column(a, b)` | Per-row comparisons |
| `eq_columns(a, b)` / `ne_columns(a, b)` | Per-row equality / inequality |
| `ordered(columns, strict)` | Per-row chain ordering |
| `sum_columns(addends, target, tolerance=None)` | Per-row sum equality |
| `unique_together(columns)` | Multi-column uniqueness |
| `implies(left, by)` | Presence implication |
| `determines(a, b)` | Functional dependency |
| `within(left, right, max_gap)` | Temporal gap bound |

See [Table checks](table-checks.md) for full signatures and semantics.

## Functions

| Function | Summary |
|---|---|
| [`infer_schema(df)`](functions.md#framelaw.infer_schema) | Draft a schema from a DataFrame or LazyFrame |
| [`sample(schema, n=10, seed=None)`](functions.md#framelaw.sample) | Generate a valid frame from a schema |
| [`framelaw.__version__`](functions.md#framelaw__version__) | Installed distribution version |

## Deprecated aliases

The flat top-level names from before 0.3 (`fl.gt`, `fl.lt_column`, …) still
work but emit `DeprecationWarning`. They forward to the scoped namespaces:

```python
fl.gt(0)                 # deprecated; use fl.checks.column.gt(0)
fl.lt_column("a", "b")   # deprecated; use fl.checks.table.lt_column("a", "b")
```

`infer_schema` and `sample` were always top-level and remain there.

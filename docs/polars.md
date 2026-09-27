# Polars integration

FrameLaw is Polars-native: validation runs on Polars' columnar memory with no
conversion step, and every native dtype can appear in a schema.

## DataFrames and LazyFrames

Both are accepted; lazy frames are collected once at the validation boundary,
so validation slots directly into lazy pipelines:

```python
schema.validate(pl.read_csv("users.csv"))
schema.validate(pl.scan_csv("users.csv"))  # LazyFrame works too
```

## Dtype support

| Polars dtype | Support |
|---|---|
| `Int8/16/32/64`, `UInt8/16/32/64` | full numeric checks |
| `Float32`, `Float64` | full numeric checks |
| `String` | regex, `one_of`, uniqueness |
| `Boolean` | nullability, uniqueness |
| `Date`, `Datetime`, `Duration`, `Time` | temporal bounds via `gt`/`lt` |
| `List`, `Array`, `Struct` | nested validation (one level) |
| `Null` | treated as always-nullable |
| other scalar dtypes | dtype + nullability enforcement |

`Column` requires an exact dtype match (structural for nested types): an
`Int32` column does not satisfy an `Int64` spec — the mismatch is reported as
a `Dtype` error and value checks are skipped for that column. A fully-null
column is accepted by any nullable spec.

## Deriving from a Polars schema

```python
schema = fl.Schema.from_polars_schema(
    {"age": pl.Int32, "email": pl.String, "score": pl.Float64}
)
```

This creates a schema with no value checks — a typed starting point you can
extend with `add_check()` or replace via inference.

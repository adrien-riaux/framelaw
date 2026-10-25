# Schemas & Columns

A `Schema` is a named collection of `Column` specs plus optional table-level
checks. It is the single object you define, persist, infer, and validate with.

## Defining a schema

```python
import polars as pl
import framelaw as fl

schema = fl.Schema(
    {
        "user_id": fl.Column(pl.Int64, nullable=False, checks=[fl.checks.column.unique()]),
        "age": fl.Column(pl.Int32, nullable=False, checks=[fl.checks.column.gt(0), fl.checks.column.lt(120)]),
        "email": fl.Column(pl.String, checks=[fl.checks.column.matches(r".+@.+\..+")]),
        "score": fl.Column(pl.Float64, checks=[fl.checks.column.between(0.0, 1.0)]),
    },
    lazy=False,  # optional, see Lazy validation
)
```

Constructor signature:

```text
Schema(columns, dataframe_checks=None, lazy=False)
```

- `columns`: dict mapping column names to `Column` objects.
- `dataframe_checks`: list of cross-column checks (see
  [Cross-column checks](cross-column-checks.md)).
- `lazy`: when `True`, `validate()` returns a report instead of raising
  (see [Lazy validation](lazy-validation.md)).

## Columns

```text
Column(dtype, nullable=True, checks=None)
```

- `dtype`: any Polars data type, e.g. `pl.Int64`, `pl.String`, `pl.Date`,
  `pl.List(pl.Int64)`, `pl.Struct({"zip": pl.String})`. Validation requires
  an exact (structural) dtype match — see [Polars integration](polars.md).
- `nullable`: when `False`, any null value fails with a `NotNull` error
  listing the offending rows. Defaults to `True`.
- `checks`: list of checks such as `fl.checks.column.gt(0)`. See [Checks](checks.md).

### Refining a column after creation

Indexing a schema with a column name returns a mutable reference:

```python
schema["age"].add_check(fl.checks.column.lt(120))
schema["email"].add_check(fl.checks.column.matches(r".+@.+\..+"))
```

Accessing a missing column raises `KeyError`. This is the intended workflow
after [inference](inference.md): infer a draft, then tighten it.

## Validating

```python
df = pl.DataFrame(
    {
        "user_id": [1],
        "age": [30],
        "email": ["a@example.com"],
        "score": [0.5],
    },
    schema={
        "user_id": pl.Int64,
        "age": pl.Int32,
        "email": pl.String,
        "score": pl.Float64,
    },
)

schema.validate(df)                        # None if valid, else raises ValueError
schema.validate(df, raise_on_error=False)  # None if valid, else ValidationReport
schema.validate_lazy(df)                   # never raises: None or ValidationReport
```

Both `polars.DataFrame` and `polars.LazyFrame` are accepted; lazy frames are
collected internally before validation.

## Alternative constructors

```python
typed = fl.Schema.from_polars_schema({"age": pl.Int64, "email": pl.String})
```

Other constructors are covered in their own guides:

- `fl.Schema.from_pydantic(Model)` — [Pydantic integration](pydantic.md).
- `fl.Schema.from_json(payload)` — [Serialization](serialization.md).
- `fl.infer_schema(df)` — [Schema inference](inference.md).

## Sampling test data

Any schema can generate valid example rows (see
[Data generation](data-generation.md)):

```python
frame = schema.sample(100, seed=42)  # deterministic with a seed
frame = fl.sample(schema, 100, seed=42)
```

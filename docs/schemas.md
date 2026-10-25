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
        "score": fl.Column(pl.Float64, checks=[fl.checks.column.between(0.0, 1.0)]),
    },
    dataframe_checks=[fl.checks.table.lt_column("start", "end")],  # optional
    lazy=False,  # optional, see Lazy validation
)
```

Constructor signature:

```python
Schema(columns, dataframe_checks=None, lazy=False)
```

- `columns`: dict mapping column names to `Column` objects.
- `dataframe_checks`: list of cross-column checks (see
  [Cross-column checks](cross-column-checks.md)).
- `lazy`: when `True`, `validate()` returns a report instead of raising
  (see [Lazy validation](lazy-validation.md)).

## Columns

```python
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
schema.validate(df)                        # None if valid, else raises ValueError
schema.validate(df, raise_on_error=False)  # None if valid, else ValidationReport
schema.validate_lazy(df)                   # never raises: None or ValidationReport
```

Both `polars.DataFrame` and `polars.LazyFrame` are accepted; lazy frames are
collected internally before validation.

## Alternative constructors

```python
Schema.from_polars_schema({"age": pl.Int64, "email": pl.String})
Schema.from_pydantic(UserModel)   # see Pydantic integration
Schema.from_json(payload)         # see Serialization
fl.infer_schema(df)               # see Schema inference
```

## Sampling test data

Any schema can generate valid example rows (see
[Data generation](data-generation.md)):

```python
frame = schema.sample(100, seed=42)  # deterministic with a seed
frame = fl.sample(schema, 100, seed=42)
```

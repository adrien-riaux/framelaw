# Decorator

`fl.guard` attaches schema validation to a Python function: the
DataFrame arguments you name are validated before the function runs, and
optionally the returned DataFrame is validated before the caller sees it.
Framelaw is Polars-native, so the check runs in Rust directly on the
frame's columns — no row-by-row conversion.

```python
import polars as pl
import framelaw as fl

users = fl.Schema(
    {
        "user_id": fl.Column(pl.Int64, nullable=False, checks=[fl.checks.column.gt(0)]),
        "email": fl.Column(pl.String, checks=[fl.checks.column.matches(r".+@.+\..+")]),
    }
)

@fl.guard(users, df="df")
def process(df: pl.DataFrame) -> pl.DataFrame:
    return df.drop_nulls()

process(pl.DataFrame({"user_id": [1, 2], "email": ["a@ex.com", "b@ex.com"]}))
```

If any call passes an invalid frame, the function is never executed and a
`fl.SchemaError` (a `ValueError` subclass) is raised with the formatted
report:

```text
Argument 'df' failed schema validation:
Validation failed with 1 error
- Column 'email' (rows 1): does not match pattern '.+@.+\..+' (first offender: 'x') [matches]
```

## Choosing the argument

Pass the parameter name(s) to `df=`:

```python
@fl.guard(users, df="orders")            # single DataFrame argument
def ingest(orders: pl.DataFrame) -> None: ...

@fl.guard(users, df=["orders", "bids"])  # several frames, same schema
def join(orders: pl.DataFrame, bids: pl.DataFrame) -> pl.DataFrame: ...
```

Keyword calls are supported, and `self` / `cls` are never touched:

```python
class Pipeline:
    @fl.guard(users, df="df")
    def run(self, df: pl.DataFrame) -> pl.DataFrame: ...

    @staticmethod
    @fl.guard(users, df="df")            # decorate below @staticmethod
    def helper(df: pl.DataFrame) -> pl.DataFrame: ...
```

Unknown parameter names fail immediately at decoration time with a
`ValueError` listing the real parameters.

## Validating the return value

Pass a second schema with `returns=`:

```python
@fl.guard(users, df="df", returns=users)
def transform(df: pl.DataFrame) -> pl.DataFrame:
    return df  # must still satisfy `users` when handed back
```

Outgoing failures raise `fl.SchemaError` starting with
`Return value failed schema validation:`.

## Options

| Argument | Meaning |
|---|---|
| `schema` | Schema for the named argument(s) |
| `df` | Parameter name or list of names (default `"df"`) |
| `returns` | Schema applied to the return value |
| `chunk_rows` | Batch size forwarded to `Schema.validate` for streaming validation |

Async functions are supported — the decorator wraps the coroutine and
validates the resolved returning frame. Passing a non-`DataFrame` value
(which includes strings or numbers) to a guarded argument raises the
usual `TypeError` from `Schema.validate`.

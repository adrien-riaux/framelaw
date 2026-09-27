# Quickstart

## Install

FrameLaw's Python package is built from Rust with
[maturin](https://www.maturin.rs/), so you need a Rust toolchain installed.
Then, from the repository root:

```bash
uv pip install -e ./framelaw-py
```

This builds the native extension and installs the `framelaw` package alongside
its Python dependencies (`polars>=0.20`, `pydantic>=2`).

Requires Python 3.10+.

## Validate your first DataFrame

```python
import polars as pl
import framelaw as fl

schema = fl.Schema(
    {
        "user_id": fl.Column(pl.Int64, nullable=False, checks=[fl.unique()]),
        "age": fl.Column(pl.Int64, nullable=False, checks=[fl.gt(0), fl.lt(120)]),
        "email": fl.Column(pl.String, checks=[fl.matches(r".+@.+\..+")]),
        "country": fl.Column(pl.String, checks=[fl.one_of(["FR", "US", "DE"])]),
    }
)

df = pl.DataFrame(
    {
        "user_id": [1, 2, 3],
        "age": [28, -4, 35],
        "email": ["a@example.com", "b@example.com", "not-an-email"],
        "country": ["FR", "US", "XX"],
    }
)

# Option A: raise on failure (default)
try:
    schema.validate(df)
except ValueError as err:
    print(err)

# Option B: get a report object instead
report = schema.validate(df, raise_on_error=False)
if not report.is_valid():
    print(report.summary())
    print(report.to_dataframe())
```

Output of `report.summary()` looks like:

```text
Column 'age': rows [1], check 'GreaterThanF64': Values in 'age' must be > 0
Column 'email': rows [2], check 'MatchesRegex': Values in 'email' do not match pattern ...
Column 'country': rows [2], check 'OneOfString': Values in 'country' are not part of the allowed set
```

## Where to go next

- New schemas from existing data? See [Schema inference](inference.md).
- Already model data with Pydantic? See
  [Pydantic integration](pydantic.md) — one line gives you a schema.
- Rules like "start must precede end"? See
  [Cross-column checks](cross-column-checks.md).
- Prefer reports over exceptions? See [Lazy validation](lazy-validation.md).
- Need test fixtures? See [Data generation](data-generation.md).

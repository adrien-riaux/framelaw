# FrameLaw

**The law your DataFrames must obey.** Native Rust schema contracts, built for
Polars, exposed to Python.

FrameLaw validates Polars `DataFrame`s against a schema — column types,
nullability, value constraints, and cross-column rules — in parallel Rust,
behind a Pandera-like Python API. On a 2M-row mixed workload it validates at
≈30M rows/s (Apple Silicon; reproduce with `examples/benchmark.py` on a
`make build-release` install).

## Features

- **Single-column checks**: nullability, uniqueness, numeric comparisons,
  ranges, regex, membership, plus `List`/`Array` lengths and `Struct`
  subfields.
- **Cross-column checks**: `start < end`-style comparisons and
  multi-column uniqueness.
- **Schema inference**: draft a schema from trusted data, then tighten it.
- **Lazy validation**: reports instead of exceptions, per call or per schema.
- **Data generation**: sample valid, reproducible test rows from any schema.
- **Pydantic v2**: derive a schema from a `BaseModel` in one line.
- **Serialization**: versioned JSON round-trips for schemas.

## Install

Requires Python 3.10+ and a Rust toolchain. From the repository root:

```bash
uv pip install -e ./framelaw-py
```

## Quickstart

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

report = schema.validate(df, raise_on_error=False)
if not report.is_valid():
    print(report.summary())
```

```text
Column 'age': rows [1], check 'GreaterThanF64': Values in 'age' must be > 0
Column 'email': rows [2], check 'MatchesRegex': Values in 'email' do not match pattern '.+@.+\..+'
Column 'country': rows [2], check 'OneOfString': Values in 'country' are not part of the allowed set
```

## Documentation

Full docs live in [`docs/`](docs/README.md): quickstart, schemas, every
check, cross-column and nested validation, error reports, inference, lazy
mode, data generation, Pydantic and Polars integration, serialization, API
reference, runnable examples, and architecture notes.

## Development

```bash
make build   # editable debug install
make test    # Rust + Python suites
make lint    # clippy, ruff, ty, interrogate
make test    # Rust + Python suites, then make examples
```

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for the full contributor workflow.

## License

MIT — see [LICENSE](LICENSE).

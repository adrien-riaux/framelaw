# Framelaw

**The law your DataFrames must obey.** Native Rust schema contracts, built for
Polars, exposed to Python.

[![PyPI](https://img.shields.io/pypi/v/framelaw.svg)](https://pypi.org/project/framelaw/)
[![Python versions](https://img.shields.io/pypi/pyversions/framelaw.svg)](https://pypi.org/project/framelaw/)
[![CI](https://github.com/adrien-riaux/framelaw/actions/workflows/ci.yml/badge.svg?branch=dev)](https://github.com/adrien-riaux/framelaw/actions/workflows/ci.yml)
[![Docs](https://img.shields.io/badge/docs-live-blue)](https://adrien-riaux.github.io/framelaw/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/adrien-riaux/framelaw/blob/main/LICENSE)

Framelaw validates Polars `DataFrame`s against a schema — column types,
nullability, value constraints, and cross-column rules — in parallel Rust,
behind a Pandera-like Python API. On a 2M-row mixed workload it validates at
≈30M rows/s (Apple Silicon; reproduce with `scripts/benchmark.py` on a
`make build-release` install).

## Features

- **Single-column checks**: nullability, uniqueness, numeric comparisons,
  ranges, regex, membership, plus `List`/`Array` lengths and `Struct`
  subfields.
- **Cross-column checks**: `start < end`-style comparisons and
  multi-column uniqueness.
- **Custom checks**: vectorized Polars expressions, batch Python callables, or
  row-wise Python predicates for domain-specific rules.
- **Schema inference**: draft a schema from trusted data, then tighten it.
- **Lazy validation**: reports instead of exceptions, per call or per schema.
- **Data generation**: sample valid, reproducible test rows from any schema.
- **Pydantic v2**: derive a schema from a `BaseModel` in one line.
- **Serialization**: versioned JSON round-trips for schemas.

## Install

```bash
uv add framelaw
```

Framelaw is published on PyPI and ships prebuilt wheels for the native Rust
extension, so no Rust toolchain is needed. It requires Python 3.10+ and pulls
in its Python dependencies (`polars>=1.0`, `pydantic>=2`) automatically.

## Quickstart

```python
import polars as pl
import framelaw as fl

schema = fl.Schema(
    {
        "user_id": fl.Column(pl.Int64, nullable=False, checks=[fl.checks.column.unique()]),
        "age": fl.Column(pl.Int64, nullable=False, checks=[fl.checks.column.gt(0), fl.checks.column.lt(120)]),
        "email": fl.Column(pl.String, checks=[fl.checks.column.matches(r".+@.+\..+")]),
        "country": fl.Column(pl.String, checks=[fl.checks.column.one_of(["FR", "US", "DE"])]),
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
if report is not None and not report.is_valid():
    print(report.summary())
```

```text
Column 'age': rows [1], check 'GreaterThanF64': Values in 'age' must be > 0
Column 'email': rows [2], check 'MatchesRegex': Values in 'email' do not match pattern '.+@.+\..+'
Column 'country': rows [2], check 'OneOfString': Values in 'country' are not part of the allowed set
```

## Documentation

Full documentation is published at
**[github.io/framelaw/](https://adrien-riaux.github.io/framelaw/)**.
It covers the quickstart, schemas, every check, cross-column and nested
validation, error reports, inference, lazy mode, data generation, Pydantic and
Polars integration, serialization, and the generated API reference.

The sources live in [`docs/`](docs/index.md) and are built with MkDocs
Material. To preview them locally:

```bash
make docs-serve   # http://127.0.0.1:8000
make docs-build   # strict build into ./site
make docs-check   # strict build + execute runnable snippets
```

## Development

```bash
make build     # editable debug install
make test      # Rust + Python suites
make lint      # clippy, ruff, ty, interrogate
make scripts   # smoke-test every script
make docs-build # strict docs build
```

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for the full contributor workflow.

## License

MIT — see [LICENSE](LICENSE).

# Framelaw

**The law your DataFrames must obey.** Native Rust schema validation for
Polars, with a Pandera-like Python API.

[![PyPI](https://img.shields.io/pypi/v/framelaw.svg)](https://pypi.org/project/framelaw/)
[![Python versions](https://img.shields.io/pypi/pyversions/framelaw.svg)](https://pypi.org/project/framelaw/)
[![CI](https://github.com/adrien-riaux/framelaw/actions/workflows/ci.yml/badge.svg?branch=dev)](https://github.com/adrien-riaux/framelaw/actions/workflows/ci.yml)
[![Docs](https://img.shields.io/badge/docs-live-blue)](https://adrien-riaux.github.io/framelaw/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/adrien-riaux/framelaw/blob/main/LICENSE)

Framelaw validates Polars `DataFrame`s and `LazyFrame`s against a declarative
schema — column types, nullability, value constraints, and cross-column rules
— in parallel Rust, behind a small Python API.

- **Rust-fast validation** on Polars' native columnar memory, with checks
  parallelized across columns.
- **Single-column checks**: nullability, uniqueness, numeric and temporal
  bounds, regex, string length, membership, list lengths, and struct fields.
- **Cross-column checks**: comparisons, per-row ordering and sums,
  multi-column uniqueness, presence implications, functional dependencies,
  and temporal gaps.
- **Custom checks**: vectorized Polars expressions, batch Python callables, or
  row-wise Python predicates, with bounded failure reports.
- **Structured failures** with the column(s), public check name, failing row
  indices, and a message; convert a report to a Polars frame to quarantine
  rows. Failures raise `fl.SchemaError` (a `ValueError` subclass).
- **Function decorator** — `@fl.guard(schema, df="df")` validates DataFrame
  arguments (and optionally the return value) of any function or method.
- **Schema inference** from trusted data, **Pydantic v2** derivation, and
  versioned **JSON** round-trips.
- **Reproducible fixtures** with seeded sampling from any schema.

## Install

```bash
pip install framelaw
```

or with [uv](https://docs.astral.sh/uv/):

```bash
uv add framelaw
```

Prebuilt wheels ship for macOS (arm64 and x86_64), Linux (x86_64 and
aarch64), and Windows (x86_64). Framelaw requires Python 3.10+, Polars 1.0+,
and Pydantic 2.x.

## Quickstart

```python
import polars as pl
import framelaw as fl

schema = fl.Schema(
    {
        "user_id": fl.Column(pl.Int64, nullable=False, checks=[fl.checks.column.unique()]),
        "age": fl.Column(
            pl.Int64, nullable=False, checks=[fl.checks.column.gt(0), fl.checks.column.lt(120)]
        ),
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
if report is not None:
    print(report.summary())
    print(report.to_dataframe())
```

`validate()` returns `None` for a valid frame and raises `ValueError` on
failure by default. Pass `raise_on_error=False` (or create the schema with
`lazy=True`) to get a `ValidationReport` instead.

## Documentation

Full documentation, guides, and the generated API reference:

**https://adrien-riaux.github.io/framelaw/**

Useful entry points:

- [Quickstart](https://adrien-riaux.github.io/framelaw/quickstart/)
- [Checks](https://adrien-riaux.github.io/framelaw/checks/)
- [Cross-column checks](https://adrien-riaux.github.io/framelaw/cross-column-checks/)
- [Decorator](https://adrien-riaux.github.io/framelaw/decorator/)
- [Error reports](https://adrien-riaux.github.io/framelaw/error-reports/)
- [Pydantic integration](https://adrien-riaux.github.io/framelaw/pydantic/)
- [API reference](https://adrien-riaux.github.io/framelaw/reference/)
- [Changelog](https://github.com/adrien-riaux/framelaw/blob/main/CHANGELOG.md)

## Development

```bash
git clone https://github.com/adrien-riaux/framelaw.git
cd framelaw
uv venv
make build      # editable debug install
make test       # Rust + Python suites
make lint       # clippy, ruff, ty, interrogate
make docs-build # strict documentation build
```

Contributions are welcome. See
[CONTRIBUTING.md](https://github.com/adrien-riaux/framelaw/blob/main/CONTRIBUTING.md)
for the full workflow.

## License

MIT — see [LICENSE](https://github.com/adrien-riaux/framelaw/blob/main/LICENSE).

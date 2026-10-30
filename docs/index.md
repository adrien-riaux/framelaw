---
description: Native Rust schema validation for Polars DataFrames.
---

<div class="hero" markdown>

# Framelaw

Validate Polars `DataFrame`s and `LazyFrame`s against a declarative schema —
types, nullability, value checks, and cross-column rules — in parallel Rust,
behind a small, Pandera-like Python API.

[Get started](quickstart.md){ .md-button .md-button--primary }
[API reference](reference/index.md){ .md-button }

</div>

<div class="hero-badges" markdown>

[![PyPI](https://img.shields.io/pypi/v/framelaw.svg)](https://pypi.org/project/framelaw/)
[![Python versions](https://img.shields.io/pypi/pyversions/framelaw.svg)](https://pypi.org/project/framelaw/)
[![CI](https://github.com/adrien-riaux/framelaw/actions/workflows/ci.yml/badge.svg?branch=dev)](https://github.com/adrien-riaux/framelaw/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/adrien-riaux/framelaw/blob/main/LICENSE)

</div>

## Why Framelaw

<div class="grid cards" markdown>

-   :material-language-rust:{ .lg .middle } **Rust-fast validation**

    ---

    Validation runs directly on Polars' columnar memory with Rayon
    parallelism. No Python-level row loops, no frame conversion.

-   :material-shield-check:{ .lg .middle } **Declarative contracts**

    ---

    Schemas are data: column specs, checks, and table rules. Persist them as
    versioned JSON, infer a draft from trusted data, or derive one from
    Pydantic v2.

-   :material-table-large:{ .lg .middle } **Column and table checks**

    ---

    Nullability, uniqueness, ranges, regex, membership, list lengths, struct
    fields, plus cross-column comparisons, sums, ordering, and dependencies.

-   :material-file-document-check:{ .lg .middle } **Structured failures**

    ---

    Every failure carries the column(s), check name, failing row indices, and
    a message. Turn a report into a Polars frame and quarantine bad rows.

-   :material-test-tube:{ .lg .middle } **Valid test data**

    ---

    Generate reproducible sample frames from any schema with a seed. The
    generated data satisfies the same checks validation enforces.

-   :material-layers-triple:{ .lg .middle } **Nested types**

    ---

    Validate `List`, `Array`, and `Struct` columns structurally, to arbitrary
    depth, including length and per-field checks.

</div>

## Install

```bash
uv add framelaw
```

or with pip:

```bash
pip install framelaw
```

Prebuilt wheels ship for macOS, Linux, and Windows, so no Rust toolchain is
needed. Framelaw requires Python 3.10+, `polars>=1.0`, and `pydantic>=2`.
See [Installation](installation.md) for source builds and platform details.

## Validate your first frame

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
    print(report.to_dataframe())
```

Output:

```text
Column 'age': rows [1], check 'GreaterThanF64': Values in 'age' must be > 0
Column 'email': rows [2], check 'MatchesRegex': Values in 'email' do not match pattern '.+@.+\..+'
Column 'country': rows [2], check 'OneOfString': Values in 'country' are not part of the allowed set
```

## Explore the docs

- **New here?** Start with the [Quickstart](quickstart.md), then read
  [Schemas & columns](schemas.md).
- **Writing rules?** Browse [Checks](checks.md) and
  [Cross-column checks](cross-column-checks.md).
- **Feeding pipelines?** See [Lazy validation](lazy-validation.md),
  [Error reports](error-reports.md), and [Recipes](recipes.md).
- **Coming from Pydantic?** Derive a schema in one line with
  [Pydantic integration](pydantic.md).
- **Looking for a specific symbol?** The [API reference](reference/index.md)
  is generated from the type stubs.

# FrameLaw Documentation

**The law your DataFrames must obey.** Native Rust schema contracts, built for
Polars, exposed to Python.

FrameLaw validates Polars `DataFrame`s (and `LazyFrame`s) against a
user-defined schema: column types, nullability, value constraints, and
cross-column rules. Validation runs in Rust with Rayon parallelism, so it stays
fast on large frames. A clean, Pandera-like Python API sits on top via PyO3.

## Contents

Start here:

- [Quickstart](quickstart.md) — install FrameLaw and validate your first
  DataFrame in five minutes.

Core concepts:

- [Schemas & Columns](schemas.md) — defining `Schema` and `Column`.
- [Checks](checks.md) — every single-column check, with examples.
- [Cross-column checks](cross-column-checks.md) — rules spanning columns
  (`start < end`, uniqueness across columns).
- [Nested types](nested-types.md) — validating `List`, `Array`, and `Struct`
  columns.
- [Error reports](error-reports.md) — reading `ValidationReport` and
  `ValidationError`.

Workflows:

- [Schema inference](inference.md) — derive a starting schema from data.
- [Lazy validation](lazy-validation.md) — reports instead of exceptions.
- [Data generation](data-generation.md) — sample valid test data from a schema.
- [Pydantic integration](pydantic.md) — derive schemas from `BaseModel`.
- [Polars integration](polars.md) — dtypes, `LazyFrame`, `from_polars_schema`.
- [Serialization](serialization.md) — persist schemas to/from JSON.

Reference & project:

- [API reference](api-reference.md) — complete class and function reference.
- [Examples](examples.md) — walkthrough of the runnable scripts in
  `examples/`.
- [Architecture](architecture.md) — how the Rust workspace is organized.
- [Changelog](../CHANGELOG.md) — release history.

# Changelog

All notable changes to Framelaw are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Python API

- `framelaw.__version__` exposes the installed distribution version.

### Packaging

- Prebuilt wheels are now published for macOS (arm64 and x86_64), Linux
  (x86_64 and aarch64), and Windows (x86_64). Installing with `pip`/`uv` no
  longer compiles Polars from source on those platforms.
- The extension targets the CPython limited API (`abi3-py310`): a single
  wheel per platform supports CPython 3.10 and newer, replacing the
  per-interpreter wheels.

## [0.1.0] — First release

Polars-native DataFrame validation in Rust, exposed to Python via PyO3.
Validation runs directly on Polars' columnar memory with Rayon parallelism
across columns and table checks.

### Validation core

- `Schema` of `ColumnSpec { name, dtype, nullable, checks }` with
  `validate()` parallelized over columns via Rayon.
- Single-column checks: `not_null`, `unique`, `gt`, `lt`, `gte`, `lte`,
  `between`, `matches` (regex), `one_of`.
- Structured errors: each `ValidationError` carries `columns`, `check`,
  failing `rows`, and a message. `ValidationReport` offers `is_valid()`,
  `summary()`, `to_dataframe()` (`column`, `columns`, `check`, `row`,
  `message`), and `raise_if_invalid()`.

### Cross-column checks

- Functional table checks: `gt_column`, `lt_column`, `gte_column`,
  `lte_column`, `eq_columns`, `ne_columns`, `unique_together`.
- Passed as `Schema({...}, dataframe_checks=[...])`; nulls on either side
  skip the row; missing columns report `ColumnExists`.

### Nested types

- Structural dtype matching for `List`, `Array`, and `Struct` (inner types
  must match, not just the outer container).
- Nested checks: `list_len_between`, `list_not_empty`,
  `struct_field(name, checks)` with one nesting level.

### Schema inference

- `infer_schema(df)` generates a starting schema in one parallel pass:
  dtype, nullability, numeric min/max bounds, string patterns
  (email, URL, ISO date, UUID), low-cardinality `one_of`, `unique`, and
  observed list-length bounds.

### Lazy validation mode

- `Schema(..., lazy=False)`; lazy schemas return a report from `validate()`
  instead of raising. Explicit `raise_on_error` always wins.
- `validate_lazy(df)` never raises; `report.raise_if_invalid()` re-raises
  on demand. The `lazy` flag round-trips through JSON.

### Data generation

- `schema.sample(n=10, seed=None)` and `fl.sample(schema, n, seed)` generate
  valid DataFrames from a schema with a seeded RNG: numeric bounds,
  `one_of`, known string patterns, list lengths, struct fields, and
  cross-column constraints. Contradictory schemas return an error.

### Compatibility

- Full numeric dtype coverage: every signed/unsigned integer width
  (including `Int128`/`UInt128`, compared exactly) and
  `Float16`/`Float32`/`Float64`.
- Accepts Polars `DataFrame` and `LazyFrame` (collected internally);
  `Schema.from_polars_schema()` derives from a native Polars schema dict.
- Pydantic v2 `Schema.from_pydantic(Model)`: `int`/`float`/`str`/`bool`,
  `Optional[T]` nullability, `gt/lt/ge/le/pattern` constraints,
  `Literal` membership, and `list[T]` with `min_length`/`max_length`.
- Schema JSON serialization (v2, loads v1): dtypes including temporal,
  `List`/`Array`/`Struct` tags, every check kind, table checks, and the
  `lazy` flag.

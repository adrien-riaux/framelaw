# framelaw
### Architecture & Design Specification
*v0.1 — Draft*

> **The law your DataFrames must obey.** Native Rust schema contracts, built for Polars, exposed to Python via PyO3.

---

## 1. Overview

framelaw is a Rust library that provides fast, expressive, schema-based validation for Polars DataFrames. It is exposed to Python via PyO3 and maturin, making it a drop-in complement to any Polars-based data pipeline.

Unlike Pandera, which runs on top of pandas and has only experimental Polars support, framelaw is Polars-native from the ground up. Validation runs directly on Polars' columnar memory layout with zero conversion overhead.

### 1.1 Goals

- Validate Polars DataFrames against a user-defined schema: types, nullability, constraints, and cross-column rules.
- Infer a schema automatically from an existing DataFrame, then let the user refine it.
- Run validation in parallel across columns using Rayon.
- Produce rich, structured error reports: which column, which row, which rule failed.
- Expose a clean, Pythonic API via PyO3 that feels familiar to Pandera users.
- **Full Polars compatibility**: support `DataFrame`, `LazyFrame`, and all native Polars dtypes.
- **Full Pydantic compatibility**: derive a framelaw `Schema` directly from any Pydantic `BaseModel` — if you already model your data with Pydantic, you get DataFrame validation for free.

### 1.2 Non-Goals

- Not a replacement for Polars itself — we validate, we don't transform.
- No async runtime required — this is a pure, synchronous validation engine.
- No YAML/JSON schema serialization in v0.1 — that comes later.

> **Why Rust over Python?**
> Validation is a hot loop: for each column, for each row, apply a check. Python's GIL prevents true parallelism here. Rust + Rayon gives us free, safe multi-threading, making framelaw significantly faster than a pure-Python equivalent on large DataFrames.

---

## 2. High-Level Architecture

The library is organized as a Cargo workspace with two tight layers: the pure Rust core (the engine) and the Python bindings (the PyO3 API). 

| Layer | Technology | Responsibility |
|---|---|---|
| Rust Core | polars, rayon, regex, jiter | Schema definition, all validation logic, error reporting, fast JSON parsing |
| Python Bindings | PyO3, maturin, pydantic-core | Expose API, zero-overhead native Pydantic schema ingest |
| Python API | Python | User-facing Schema, Column, Check classes |
| Polars Compat | polars Python API | LazyFrame support, dtype bridge, Polars schema interop |
| Pydantic Compat | pydantic v2 | BaseModel → Schema derivation, field type mapping |

### 2.1 Module Structure

To guarantee a clean, decoupled architecture, the project is structured as a Cargo workspace with two distinct crates. This isolates Python-specific noise from core validation logic.

```
framelaw/
├── Cargo.toml                  ← Workspace manifest
├── framelaw-core/              ← Pure Rust validation engine
│   ├── Cargo.toml              ← polars, rayon, regex, jiter dependencies
│   └── src/
│       ├── schema.rs           ← Schema struct + validate() orchestration
│       ├── column.rs           ← ColumnSpec: name, dtype, nullable, checks
│       ├── checks.rs           ← Check trait + all built-in check implementations
│       ├── inference.rs        ← Schema inference from a live DataFrame
│       └── errors.rs           ← ValidationError and ValidationReport types
├── framelaw-py/                ← Python FFI & PyO3 bindings
│   ├── Cargo.toml              ← pyo3, pydantic-core dependencies
│   ├── pyproject.toml          ← maturin build config
│   ├── src/
│   │   ├── lib.rs              ← PyO3 module entry point
│   │   └── pydantic_compat.rs  ← Deep pydantic-core interop via Rust
│   └── python/
│       └── framelaw/
│           ├── compat/
│           │   ├── polars.py   ← LazyFrame support, Polars schema interop
│           │   └── pydantic.py ← BaseModel → framelaw Schema derivation
│           └── __init__.py
```

### 2.2 Code Quality Standards

A robust architecture is maintained by strict tooling enforcement:
- **Rust Formatting & Linting**: Enforced via strict `rustfmt` and aggressive `clippy` configurations to ensure idiomatically sound, allocation-wary, and zero-cost code where possible.
- **Python Formatting**: Enforced via `ruff`.
- **Testing**: Pure Rust unit tests inside `framelaw-core`, and broader PyTest integration tests testing the API wrapper in the project root.


---

## 3. Core Components

### 3.1 The Check Trait  (checks.rs)

Every validation rule implements a single Rust trait. This is the central abstraction of the library.

```rust
pub trait Check: Send + Sync {
    fn name(&self) -> &str;
    fn validate(&self, series: &Series) -> Vec<ValidationError>;
}
```

The `Send + Sync` bounds are required for Rayon parallelism. Every built-in check is a struct that implements this trait.

#### Built-in Checks (v0.1)

| Check | Python API | Description |
|---|---|---|
| NotNull | `pv.not_null()` | Fails if any value in the column is null |
| GreaterThan | `pv.gt(n)` | Fails if any value ≤ n |
| LessThan | `pv.lt(n)` | Fails if any value ≥ n |
| Between | `pv.between(lo, hi)` | Fails if any value is outside [lo, hi] |
| Matches | `pv.matches(r"regex")` | Fails if any string value does not match the pattern |
| OneOf | `pv.one_of(["a","b"])` | Fails if any value is not in the allowed set |
| UniqueValues | `pv.unique()` | Fails if the column contains duplicate values |

### 3.2 ColumnSpec  (column.rs)

A `ColumnSpec` describes a single column: its expected name, Polars data type, whether nulls are allowed, and the list of checks to run against it.

```rust
pub struct ColumnSpec {
    pub name: String,
    pub dtype: DataType,       // Polars DataType
    pub nullable: bool,
    pub checks: Vec<Box<dyn Check>>,
}
```

### 3.3 Schema  (schema.rs)

The `Schema` holds a collection of `ColumnSpec`s and owns the `validate()` method. Validation is parallelized across columns using Rayon.

```rust
pub struct Schema {
    columns: Vec<ColumnSpec>,
}

impl Schema {
    pub fn validate(&self, df: &DataFrame) -> ValidationReport {
        let errors: Vec<ValidationError> = self.columns
            .par_iter()                          // Rayon parallel iterator
            .flat_map(|col_spec| col_spec.validate(df))
            .collect();
        ValidationReport { errors }
    }
}
```

### 3.4 ValidationReport & ValidationError  (errors.rs)

Errors are structured data, not just strings. Each error carries the column name, the failing row indices, and the name of the check that failed.

```rust
pub struct ValidationError {
    pub column: String,
    pub check:  String,
    pub rows:   Vec<usize>,    // indices of failing rows
    pub message: String,
}

pub struct ValidationReport {
    pub errors: Vec<ValidationError>,
}

impl ValidationReport {
    pub fn is_valid(&self) -> bool { self.errors.is_empty() }
    pub fn raise_if_invalid(&self) -> PyResult<()> { ... }
}
```

---

## 4. Schema Inference  (inference.rs)

One of the most useful features: given an existing DataFrame, framelaw can automatically infer a starting `Schema` from it. The user can then review, tighten, and persist that schema.

> **The Core Idea:** Inference is a first draft, not a final answer. It observes what the data looks like today and generates conservative rules. The user is expected to review and strengthen the output.

### 4.1 What Gets Inferred

| Property | How it is inferred |
|---|---|
| `dtype` | Directly from the Polars Series dtype |
| `nullable` | `true` if the column contains at least one null value |
| `gt` / `lt` bounds | Observed min and max values (with a small margin) |
| `matches` pattern | Common patterns detected: email, URL, ISO date, UUID |
| `one_of` | If unique value count ≤ 20, treat as categorical and list them |
| `unique` | Flagged if the column has no duplicates in the sample |

### 4.2 Python API for Inference

```python
import framelaw as fl
import polars as pl

df = pl.read_csv("users.csv")

# Infer schema from the DataFrame
schema = fl.infer_schema(df)

# Inspect what was inferred
print(schema)  # human-readable summary

# Refine: tighten a bound, add a regex check
schema["age"].add_check(fl.lt(120))
schema["email"].add_check(fl.matches(r".+@.+\..+"))

# Now validate
schema.validate(df)   # raises ValidationError if anything fails
```

### 4.3 Inference Algorithm

Inference runs in a single parallel pass over all columns using Rayon:

- For each column, read the Polars `DataType` → set `dtype`.
- Count nulls → set `nullable = true` if `null_count > 0`.
- For numeric columns, compute min/max → emit `gt(min - ε)` and `lt(max + ε)` as suggested checks.
- For string columns, test each value against known regex patterns (email, UUID, ISO-8601 date). If all values match, emit `matches(pattern)`.
- Count distinct values. If `distinct_count / total ≤ 0.05` and `distinct_count ≤ 20` → emit `one_of([...])`.
- If `null_count == 0` and all values are unique → emit `unique()`.

---

## 5. Python API Design

The PyO3 bindings expose four classes to Python: `Schema`, `Column`, `Check`, and `ValidationReport`. The API is intentionally Pandera-like to minimise the learning curve.

### 5.1 Full Usage Example

```python
import framelaw as fl
import polars as pl

schema = fl.Schema({
    "user_id": fl.Column(pl.Int64,   checks=[fl.not_null(), fl.unique()]),
    "age":     fl.Column(pl.Int32,   checks=[fl.gt(0), fl.lt(120)]),
    "email":   fl.Column(pl.Utf8,    checks=[fl.matches(r".+@.+\..+")]),
    "score":   fl.Column(pl.Float64, checks=[fl.between(0.0, 1.0)]),
    "country": fl.Column(pl.Utf8,    checks=[fl.one_of(["FR", "US", "DE"])]),
})

df = pl.read_csv("users.csv")

# Option A: raises ValidationError with full report
schema.validate(df)

# Option B: returns a report without raising
report = schema.validate(df, raise_on_error=False)
if not report.is_valid():
    print(report.summary())
    # → Column 'age': 3 rows failed check 'gt(0)' → rows [12, 87, 204]
```

### 5.2 ValidationReport in Python

```python
report.is_valid()       # → bool
report.summary()        # → human-readable string
report.errors           # → list of dicts: {column, check, rows, message}
report.to_dataframe()   # → pl.DataFrame of all failing rows
```

---

---

## 6. Compatibility

### 6.1 Polars Compatibility

framelaw supports all Polars usage patterns out of the box.

#### DataFrame vs LazyFrame

framelaw accepts both `DataFrame` and `LazyFrame`. When a `LazyFrame` is passed, framelaw calls `.collect()` internally before validation. This means you can slot validation directly into a lazy pipeline without breaking it.

```python
import framelaw as fl
import polars as pl

schema = fl.Schema({ ... })

# Works with LazyFrame — collect happens internally
df = pl.scan_csv("users.csv")
schema.validate(df)   # ← LazyFrame accepted directly
```

#### Polars dtype mapping

All Polars dtypes are supported as first-class citizens in `fl.Column(dtype=...)`. framelaw also accepts a native Polars schema (a `dict[str, DataType]`) and converts it directly:

```python
# Derive a framelaw Schema from a Polars schema dict
polars_schema = {"age": pl.Int32, "email": pl.Utf8, "score": pl.Float64}
schema = fl.Schema.from_polars_schema(polars_schema)
```

#### Supported Polars dtypes

| Polars dtype | Supported | Notes |
|---|---|---|
| `Int8/16/32/64`, `UInt8/16/32/64` | ✅ | Full numeric checks |
| `Float32`, `Float64` | ✅ | Full numeric checks |
| `Utf8` / `String` | ✅ | Regex, one_of, unique |
| `Boolean` | ✅ | not_null, one_of |
| `Date`, `Datetime`, `Duration` | ✅ | Temporal bounds via gt/lt |
| `List`, `Struct`, `Array` | ⚠️ v0.2 | Nested type validation planned |
| `Categorical`, `Enum` | ✅ | one_of auto-populated |
| `Null` | ✅ | Treated as always-nullable |

---

### 6.2 Pydantic Compatibility

If you already define your data models with Pydantic, you can derive a framelaw `Schema` directly from a `BaseModel` — no duplication needed.

#### Deriving a Schema from a BaseModel

```python
import framelaw as fl
from pydantic import BaseModel, Field
from typing import Optional

class User(BaseModel):
    user_id: int
    age:     int       = Field(gt=0, lt=120)
    email:   str       = Field(pattern=r".+@.+\..+")
    score:   float     = Field(ge=0.0, le=1.0)
    country: Optional[str] = None

# One line — all Field constraints are translated automatically
schema = fl.Schema.from_pydantic(User)

df = pl.read_csv("users.csv")
schema.validate(df)
```

#### Pydantic → framelaw type mapping

Unlike Python-only validation frameworks, framelaw leverages `pydantic-core` (the Rust engine powering Pydantic v2) and `jiter` (for high-performance JSON parsing). This native ingestion of Pydantic schema representations at the Rust level cuts out Python reflection overhead.

framelaw reads Pydantic field types and `Field(...)` constraints and maps them to the equivalent framelaw checks:

| Pydantic | framelaw Column dtype | framelaw Checks |
|---|---|---|
| `int` | `pl.Int64` | `not_null()` |
| `float` | `pl.Float64` | `not_null()` |
| `str` | `pl.Utf8` | `not_null()` |
| `bool` | `pl.Boolean` | `not_null()` |
| `Optional[T]` | dtype of T | `nullable = True` |
| `Field(gt=n)` | — | `gt(n)` |
| `Field(lt=n)` | — | `lt(n)` |
| `Field(ge=n, le=m)` | — | `between(n, m)` |
| `Field(pattern=r"...")` | — | `matches(r"...")` |
| `Literal["a","b"]` | — | `one_of(["a","b"])` |

> **Note on Pydantic v1 vs v2:** framelaw targets **Pydantic v2**. Pydantic v1 is supported via a thin compatibility shim that normalises `Field` metadata before processing. A deprecation warning is emitted when v1 is detected.

---

## 7. Roadmap

| Version | Feature |
|---|---|
| v0.1 | Core checks, Schema, ColumnSpec, basic Python bindings |
| v0.1 | Schema inference (`infer_schema`) |
| v0.1 | Full Polars dtype support + `LazyFrame` acceptance |
| v0.1 | Pydantic v2 `BaseModel` → `Schema` derivation via `pydantic-core` |
| v0.1 | Schema serialization to / from JSON using `jiter` |
| v0.2 | Cross-column checks (e.g. `col_a < col_b`) |
| v0.2 | Nested Polars types: `List`, `Struct`, `Array` |
| v0.3 | Lazy validation mode (returns report instead of raising by default) |
| v0.3 | Hypothesis-style data generation from schema for testing |
| v0.3 | Pydantic v1 compatibility shim removed (v1 EOL) |

# framelaw Context & Instructions

**Project Context**
framelaw is a Polars-native DataFrame validation library built in Rust and exposed to Python. It validates data in parallel against schemas, integrating smoothly with Pydantic V2. Current package version is `0.1.0` (see `framelaw-py/pyproject.toml` and both `Cargo.toml` files — keep them in sync).

**Core Architecture**
The repository is structured as a Cargo workspace with a strict separation of concerns:
1. `framelaw-core`: Pure Rust. No Python/PyO3 dependencies. Handles schema definitions, validation logic, data generation, and parallel execution.
2. `framelaw-py`: PyO3 bindings & FFI. Exposes the Python API and interfaces with `pydantic-core`.

Module map (`framelaw-core/src`):
- `schema.rs` — `Schema` struct + `validate()` orchestration (columns, then table checks).
- `column.rs` — `ColumnSpec`: name, dtype, nullable, checks; structural dtype matching.
- `checks.rs` — `Check` trait + all built-in single-column checks; `CheckSpec` enum is the serializable form of every check.
- `dataframe_checks.rs` — `DataFrameCheck` trait + cross-column checks (`ColumnComparison`, `UniqueTogether`).
- `inference.rs` — schema inference from a live DataFrame.
- `generate.rs` — valid-data sampling from a schema (seeded RNG).
- `errors.rs` — `ValidationError` (`columns`, `check`, `rows`, `message`) and `ValidationReport`.

**Key Technologies**
- **Data Engine**: `polars` (runs directly on native Rust structs, zero Python conversion overhead).
- **Concurrency**: `rayon` (parallel validation over sets of columns/rules).
- **Schema/JSON**: `pydantic-core` (Rust-native Pydantic ingestion), `jiter` (fast JSON pre-validation), `serde`/`serde_json`.
- **Python Bindings**: `pyo3`, `pyo3-polars`, and `maturin`.
- **Sampling RNG**: `rand` / `rand_chacha` (seeded, reproducible).
- **Python quality**: `ruff` (lint + format), `ty` (type checking), `interrogate` (docstring coverage, fail-under 95).

**Coding Standards**
- **Rust**: Strict `rustfmt` and aggressive `clippy` (`cargo clippy --all-targets -- -D warnings`). Keep code idiomatic and memory-efficient (zero-allocation where possible; prefer borrows/lifetimes over cloning).
- **Python**: `ruff` for formatting and linting, `ty check` clean, `interrogate` at 100% for `framelaw-py/python` and `scripts/`.
- **Type stubs**: `framelaw-py/python/framelaw/_framelaw.pyi` mirrors the native module for checkers and IDEs. Keep it in sync with `framelaw-py/src/lib.rs` whenever signatures change (runtime ignores it; `ty` requires it).
- **Testing**: Pure Rust unit tests inside `framelaw-core`. Python API integration tests using `pytest` for `framelaw-py`.

**Development Rules**
- **Maintain Separation**: NEVER leak Python types (`PyResult`, `PyAny`, `PyObject`) or PyO3 annotations into the `framelaw-core` crate.
- **Pydantic Focus**: Prioritize `pydantic-core` at the Rust layer for handling schema derivation rather than reflecting Python `BaseModel`s via Python FFI.
- **New checks checklist**: every new check needs all of: core struct + `CheckSpec` variant (+ `check_from_spec` if nestable), Python constructor in `framelaw-py/src/lib.rs`, `JsonCheck` round-trip, stub entry in `_framelaw.pyi`, `__init__.py` export, docs in `docs/checks.md` (or the relevant page), and tests on both sides.
- **JSON compatibility**: schema JSON is at v2 and loads v1. Never break loading older payloads; bump the version only with a migration path.
- **Scripts**: smoke tests and benchmarks live in `scripts/`, named by capability (never by milestone). Each is runnable (`uv run --with-editable ./framelaw-py python scripts/<name>.py`), self-asserting, with `def main() -> None`, full docstrings, and `Optional`-narrowing asserts before touching `validate()` results.
- **Docs**: live in `docs/*.md`. Every code snippet must actually execute against the built extension before committing — dtype mismatches (e.g. `Int32` vs inferred `Int64`) are the classic failure.
- **CHANGELOG**: add an entry under the unreleased section for every user-facing change.
- **Iterative Work**: Always work step-by-step. Propose and confirm architectural setup before writing large boilerplate blocks.

**Common Commands** (see `Makefile`)
- `make build` — build + install the extension (`uv pip install -e ./framelaw-py`, debug).
- `make build-release` — optimized `--release` install for benchmarks.
- `make test` — `cargo test -p framelaw-core` + pytest suite.
- `make lint` — full verify: clippy, ruff, `ty`, interrogate.
- `make test` — Rust + Python suites; `make scripts` runs every script.
- `make wheel` / `make publish` — release wheel build / TestPyPI upload.

**References**
- Start with `docs/README.md` (doc index) and `docs/architecture.md` (workspace layout, design rules).
- API facts live in `docs/api-reference.md`; keep it in sync with `framelaw-py/src/lib.rs` and `_framelaw.pyi`.

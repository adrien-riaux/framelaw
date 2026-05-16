# framelaw Context & Instructions

**Project Context**
framelaw is a Polars-native DataFrame validation library built in Rust and exposed to Python. It validates data in parallel against schemas, integrating smoothly with Pydantic V2.

**Core Architecture**
The repository is structured as a Cargo workspace with a strict separation of concerns:
1. `framelaw-core`: Pure Rust. No Python/PyO3 dependencies. Handles schema definitions, validation logic, and parallel execution.
2. `framelaw-py`: PyO3 bindings & FFI. Exposes the Python API and interfaces with `pydantic-core`.

**Key Technologies**
- **Data Engine**: `polars` (runs directly on native Rust structs, zero Python conversion overhead).
- **Concurrency**: `rayon` (parallel validation over sets of columns/rules).
- **Schema/JSON**: `pydantic-core` (Rust-native Pydantic ingestion), `jiter` (fast JSON serialization/parsing).
- **Python Bindings**: `pyo3` and `maturin`.

**Coding Standards**
- **Rust**: Strict `rustfmt` and aggressive `clippy`. Keep code idiomatic and memory-efficient (zero-allocation where possible; prefer borrows/lifetimes over cloning).
- **Python**: Use `ruff` for formatting and linting.
- **Testing**: Pure Rust unit tests inside `framelaw-core`. Python API integration tests using `pytest` for `framelaw-py`.

**Development Rules**
- **Maintain Separation**: NEVER leak Python types (`PyResult`, `PyAny`, `PyObject`) or PyO3 annotations into the `framelaw-core` crate.
- **Pydantic Focus**: Prioritize `pydantic-core` at the Rust layer for handling schema derivation rather than reflecting Python `BaseModel`s via Python FFI.
- **Iterative Work**: Always work step-by-step. Propose and confirm architectural setup before writing large boilerplate blocks.

**References**
- Always refer to `docs/framelaw-architecture-v01.md` if unsure about the design.

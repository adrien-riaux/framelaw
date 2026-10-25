# Why FrameLaw

FrameLaw is a schema-first validator for Polars. It is deliberately narrow:
one data engine, one schema representation, and a small API that is easy to
persist and reuse.

## What FrameLaw optimizes for

- **Validation speed on Polars data.** The engine runs in Rust on Polars'
  columnar buffers, with checks parallelized across columns. There is no
  conversion to Python objects or to another frame library.
- **Schemas as data.** Every rule is a serializable value. A schema can be
  written by hand, inferred from trusted data, derived from a Pydantic model,
  or loaded from JSON — and all four produce the same internal representation.
- **Actionable failures.** Validation returns structured errors with the
  failing row indices, not a bare boolean. The report can be converted to a
  Polars frame for quarantining or dashboards.
- **Reproducible fixtures.** The same schema that validates production data
  can generate seeded, valid sample frames for tests.

## How it compares

| | FrameLaw | Pandera | Great Expectations | Hand-written checks |
|---|---|---|---|---|
| Primary engine | Polars + Rust | Pandas / Polars / PySpark | Multiple backends | Whatever you write |
| Schema definition | Python objects + versioned JSON | Python / YAML | YAML / Python | Ad hoc |
| Single-column checks | Yes | Yes | Yes | Yes |
| Cross-column checks | Yes | Yes | Yes | Yes |
| Structured row-level errors | Yes | Yes | Yes | Usually not |
| Data generation from schema | Yes | Limited | No | No |
| Native extension | Yes (PyO3) | No | No | No |
| Maturity | 0.x, focused API | Mature, broad | Mature, broad | — |

The comparison is about shape, not a claim that one tool is universally
better. Pandera and Great Expectations cover more backends and more
operational features; FrameLaw trades that breadth for a tight Polars/Rust
implementation and a schema format that is trivial to serialize.

## When FrameLaw is a good fit

- Your data already lives in **Polars**, and validation is on the hot path.
- You want a **single schema artifact** that can be checked into a repo,
  loaded by a service, and used to generate test data.
- You want **row-level failure details** without writing custom masks.
- You model inputs with **Pydantic v2** and want the schema derived from the
  model rather than duplicated.

## When it is not

- You need **Pandas, Spark, or SQL** validation. FrameLaw is Polars-only.
- You need a **profiling UI, data docs, or an expectation suite runner**.
  Great Expectations is the better tool for that workflow.
- You need **streaming validation** over unbounded data. `chunk_rows` batches
  a finite frame; it is not a streaming engine.
- You need a **stable 1.0 API**. FrameLaw is pre-1.0 and may still make
  breaking changes between minor releases; see the
  [changelog](changelog.md).

## Design principles

1. **The Rust core knows nothing about Python.** `framelaw-core` is pure
   Rust; PyO3 types stay in the bindings crate.
2. **Checks are values, not closures.** Declarative rules can be serialized,
   inferred, diffed, and generated from.
3. **Collect errors, don't short-circuit.** One validation pass reports every
   problem it can.
4. **Raising is a Python convenience.** The core always returns a report; the
   bindings decide whether to raise `ValueError`.
5. **Strict dtype matching.** An `Int32` column does not satisfy an `Int64`
   spec. Structural matching for nested types keeps contracts explicit.

Ready to try it? Start with the [Quickstart](quickstart.md), or read the
[benchmarks](benchmarks.md) methodology before quoting numbers.

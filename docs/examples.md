# Examples

Runnable scripts live in `examples/`. Run any of them with:

```bash
uv run --with-editable ./framelaw-py python examples/<name>.py
```

## Start here

- `basic_validation.py` — core workflows end to end: explicit schemas,
  inference plus refinement plus JSON persistence, `LazyFrame` validation,
  Pydantic derivation, temporal columns, and reading failure reports.
- `infer_schema.py` — the infer/refine/enforce loop on its own: draft a
  schema from trusted data, tighten a bound, reject new bad rows.
- `pydantic_model.py` — derive a schema from a `BaseModel` and validate
  good and bad frames, including an optional field.
- `error_report.py` — every way to read failures: `is_valid`, `summary`,
  per-error attributes, `to_dataframe`, and `raise_if_invalid`.

## Going further

- `cross_column_nested.py` — table checks (`lt_column`, `unique_together`)
  combined with `List` length checks and `Struct` subfield validation,
  plus JSON round-trip and Pydantic `list[int]`.
- `lazy_sampling.py` — `lazy=True` schemas, `validate_lazy`,
  `raise_if_invalid`, seeded `sample()` with self-validation, and the
  top-level `fl.sample()`.
- `generate_data.py` — sampling focused: reproducible fixtures from a
  schema with scalar, categorical, and cross-column constraints.
- `benchmark.py` — validation throughput on a multi-million-row frame
  (best-of-5). Run `make build-release` first; numbers from debug builds
  are meaningless.

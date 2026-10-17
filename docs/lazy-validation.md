# Lazy validation

By default `validate()` raises `ValueError` on failure — convenient for
scripts, awkward for pipelines that want to inspect, log, or branch on
failures. Lazy mode returns reports instead.

## Per-call lazy validation

```python
report = schema.validate(df, raise_on_error=False)
if not report.is_valid():
    log(report.summary())
```

`validate_lazy()` is a shorthand that never raises, regardless of settings:

```python
report = schema.validate_lazy(df)  # None if valid, else ValidationReport
```

## Schema-level lazy mode

Pass `lazy=True` to flip the default for that schema. Explicit
`raise_on_error` still wins per call:

```python
schema = fl.Schema({...}, lazy=True)

schema.validate(df)                     # report, never raises
schema.validate(df, raise_on_error=True)  # raises again for this call
schema.lazy                             # True
```

The flag persists through `to_json()` / `from_json()`.

## Re-raising from a report

```python
report = schema.validate_lazy(df)
report.raise_if_invalid()  # ValueError with the summary, or None if valid
```

This makes the common pipeline shape explicit: validate lazily everywhere,
raise exactly where a failure must stop the run.

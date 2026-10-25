# Serialization

Schemas persist to compact JSON and load back with identical behavior —
handy for storing contracts alongside pipelines or sharing them between
services.

```python
import polars as pl
import framelaw as fl

schema = fl.Schema(
    {
        "id": fl.Column(pl.Int64, nullable=False),
        "score": fl.Column(pl.Float64, checks=[fl.checks.column.between(0.0, 1.0)]),
    }
)
df = pl.DataFrame({"id": [1, 2], "score": [0.25, 0.75]})

payload = schema.to_json()
restored = fl.Schema.from_json(payload)
assert restored.validate(df) is None
```

## What is persisted

- Every column: name, dtype (including temporal units and nested
  `List:`/`Array:`/`Struct:` tags), nullability, and all checks.
- Every table check (`lt_column`, `unique_together`, …).
- The `lazy` flag.

Malformed payloads raise `ValueError` (invalid JSON), as do unknown schema
versions and unsupported dtype tags.

## Versioning

Payloads carry a `version` field. The current writer emits version 2, and
the loader accepts versions 1 and 2, so schemas persisted by older releases
keep loading. Timezone-aware datetimes are rejected at serialization time
because their semantics don't round-trip through the current format.

# Data generation

Any schema can produce valid example rows — useful for tests, property-based
checks, and demos. Generation runs in Rust from the same checks validation
enforces, with a seeded RNG for reproducibility.

```python
frame = schema.sample(100)            # 100 valid rows, random seed
frame = schema.sample(100, seed=42)   # deterministic
frame = fl.sample(schema, 100, seed=42)  # top-level equivalent
```

The sampled frame validates clean against its own schema:

```python
assert schema.validate(frame) is None
```

## How values are chosen

- Numerics (incl. temporal): uniform within the intersection of
  `gt`/`gte`/`lt`/`lte`/`between` bounds.
- `one_of`: random pick from the allowed set.
- `matches`: canned valid values for known patterns (email, URL, ISO date,
  UUID, 5-digit zip, 2-letter code); `"valid"` otherwise.
- `unique` / `unique_together`: repaired by resampling.
- `nullable=True`: ~5% nulls injected (never for unique columns).
- Lists/arrays: lengths within the inferred or declared bounds; structs built
  field-by-field honoring `struct_field` checks.
- Cross-column comparisons: satisfied by joint rejection sampling
  (e.g. `end` is drawn above `start` for `lt_column("start", "end")`).

## Limits

- One nesting level; full regex synthesis is out of scope (unknown patterns
  fall back to `"valid"`, which may not satisfy an exotic regex).
- Contradictory schemas (e.g. `gt(10)` combined with `lt(0)`, or an
  unsatisfiable `unique`) raise `ValueError` instead of returning data.

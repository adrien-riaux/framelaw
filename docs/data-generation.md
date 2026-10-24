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
- `one_of`: random pick from the allowed set, intersected with any
  `matches` patterns (empty intersections error).
- `matches`: canned valid values for known patterns (email, URL, ISO date,
  UUID, 5-digit zip, 2-letter code). A pattern with no known canned value,
  or a set of patterns no single canned value satisfies, raises a
  cannot-synthesize error instead of emitting potentially invalid data.
- `unique`: resolved directly on small known domains (`one_of` pools are
  shuffle-taken, pattern-locked strings expanded by mutation); otherwise
  distinct values are harvested across batches. Impossible demands (e.g.
  `bool` with more than two rows) error instead of looping.
- `unique_together`: repaired by resampling.
- `nullable=True`: ~5% nulls injected (skipped when the column already holds
  distinct values, or when an explicit `not_null()` check forbids nulls).
- Lists/arrays: lengths within the inferred or declared bounds; structs built
  field-by-field honoring `struct_field` checks.
- Cross-column comparisons: satisfied by joint rejection sampling
  (e.g. `end` is drawn above `start` for `lt_column("start", "end")`).

## Limits

- Elements inside nested containers are generated without element-level
  checks (`list_len_between` still applies); struct fields keep their
  `struct_field` checks.
- Full regex synthesis is out of scope (patterns without
  a known canned value raise a cannot-synthesize error).
- Contradictory schemas (e.g. `gt(10)` combined with `lt(0)`, or an
  unsatisfiable `unique`) raise `ValueError` instead of returning data.
- Sampled list lengths are capped for materialization safety; schemas
  demanding huge lists report a generation-capacity error.

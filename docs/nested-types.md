# Nested types

Framelaw validates Polars nested columns — `List`, `Array`, and `Struct` —
with structural dtype matching and dedicated checks, nested to arbitrary
depth.

## Dtype matching

Nested dtypes must match **structurally**, not just by container:

- `pl.List(pl.Int64)` does not accept `pl.List(pl.String)`.
- `pl.Array(pl.Int64, 2)` requires the same width.
- `pl.Struct({"a": pl.Int64})` requires the same field names and inner types.

Scalar dtypes still require exact equality.

## Lists and arrays

Declare the full nested dtype and add length checks:

```python
fl.Column(
    pl.List(pl.Int64),
    checks=[fl.checks.column.list_len_between(1, 3), fl.checks.column.list_not_empty()],
)
```

- `fl.checks.column.list_len_between(min_len, max_len)` fails with the outer row indices
  whose list length falls outside the inclusive range.
- `fl.checks.column.list_not_empty()` fails on empty (non-null) lists.
- Null rows are skipped; govern them with `nullable`.

## Structs

Extract a subfield and run ordinary checks against it. Failures are reported
against the parent column with the field named in the message:

```python
fl.Column(
    pl.Struct({"city": pl.String, "zip": pl.String}),
    checks=[fl.checks.column.struct_field("zip", [fl.checks.column.matches(r"^\d{5}$")])],
)
```

Referencing a field that does not exist reports an error with an empty row
list.

## Inference and generation

- `infer_schema()` passes nested dtypes through and infers starting
  `list_len_between` bounds from the observed min/max lengths.
- `schema.sample()` generates nested values honoring the length bounds and
  struct field checks.
- Pydantic `list[T]` fields map to `pl.List` columns (see
  [Pydantic integration](pydantic.md)).

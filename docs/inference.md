# Schema inference

Given an existing DataFrame, Framelaw can draft a starting `Schema` for you.
Inference is a first draft, not a final answer: review it, tighten it, then
persist it.

```python
df = pl.read_csv("users.csv")

schema = fl.infer_schema(df)
print(schema.to_json())  # inspect what was inferred

schema["age"].add_check(fl.checks.column.lt(120))
schema["email"].add_check(fl.checks.column.matches(r".+@.+\..+"))

schema.validate(df)  # raises if anything fails
```

## What gets inferred

| Property | Rule |
|---|---|
| `dtype` | taken directly from the Polars column |
| `nullable` | `True` if the column contains any null |
| `gt` / `lt` bounds | observed min/max with a small margin (skipped for 64- and 128-bit ints that could lose precision as floats); temporal columns get the same bound in their tick unit — whole days for `Date`, nanoseconds for `Datetime`/`Time`, the duration unit for `Duration` |
| `matches` pattern | emitted when every non-null string matches a known pattern: email, URL, ISO date, UUID |
| `one_of` | emitted when a string column has ≤ 20 distinct values covering ≤ 5% of rows |
| `unique` | emitted when all values are distinct and non-null |
| list lengths | `list_len_between` from observed min/max lengths |

Inference runs in a single parallel pass over all columns. It never invents
cross-column checks.

## Typical workflow

1. `schema = fl.infer_schema(df)` on trusted data.
2. Review `schema.to_json()` and strengthen weak bounds.
3. Add checks via `schema["col"].add_check(...)`.
4. Save with `to_json()` and validate new data against the persisted schema.

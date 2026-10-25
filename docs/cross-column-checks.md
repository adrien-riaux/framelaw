# Cross-column checks

Some rules span columns — "start must precede end", "these two columns are
unique together". These are **table checks**: they see the whole frame and are
declared separately from per-column checks.

```python
schema = fl.Schema(
    {
        "start": fl.Column(pl.Int64),
        "end": fl.Column(pl.Int64),
        "country": fl.Column(pl.String),
        "city": fl.Column(pl.String),
    },
    dataframe_checks=[
        fl.checks.table.lt_column("start", "end"),
        fl.checks.table.unique_together(["country", "city"]),
    ],
)

df = pl.DataFrame(
    {
        "start": [2],
        "end": [1],
        "country": ["FR"],
        "city": ["Paris"],
    }
)
```

## Comparisons

| Function | Rule |
|---|---|
| `fl.checks.table.lt_column(a, b)` | `a < b` |
| `fl.checks.table.lte_column(a, b)` | `a <= b` |
| `fl.checks.table.gt_column(a, b)` | `a > b` |
| `fl.checks.table.gte_column(a, b)` | `a >= b` |
| `fl.checks.table.eq_columns(a, b)` | `a == b` |
| `fl.checks.table.ne_columns(a, b)` | `a != b` |
| `fl.checks.table.ordered([a, b, ...], strict)` | chain `a (<=|<) b (<=|<) ...` per row |
| `fl.checks.table.sum_columns([a, b, ...], target, tolerance=None)` | `target == a + b + ...` per row |
| `fl.checks.table.implies(left, by)` | non-null `left` requires non-null `by` |
| `fl.checks.table.determines(a, b)` | `a` functionally determines `b` |
| `fl.checks.table.within(left, right, max_gap: timedelta)` | `right - left` within `max_gap` (temporal) |

Semantics:

- Both columns must be numeric (including temporal) or both strings;
  mixing kinds reports an error.
- Temporal columns must share the exact same dtype: comparing `Date`
  (day counts) against `Datetime` (microsecond ticks) would compare
  unrelated magnitudes, so it reports an error instead of guessing.
- Rows where either side is null are **skipped** — enforce presence with
  `nullable=False` on the columns if nulls must fail.
- The two column names must differ; otherwise construction raises
  `ValueError`.
- A missing column reports a `ColumnExists` error instead of row failures.

Each failure is reported once with `columns == [a, b]` and the failing row
indices.

## Uniqueness across columns

```python
fl.checks.table.unique_together(["country", "city"])
```

Fails when the same combination appears in more than one row; all rows in
each duplicated group are reported. Nulls participate in the duplicate scan,
so repeated null combinations count as duplicates. Needs at least two
columns.

## Chain ordering across columns

```python
# a <= b <= c in every row
fl.checks.table.ordered(["a", "b", "c"], strict=False)
# a < b < c (strict mode rejects equal neighbours too)
fl.checks.table.ordered(["a", "b", "c"], strict=True)
```

- The chain is evaluated per row over adjacent pairs; each violation is
  reported with the involved columns and the failing row indices.
- All columns must be numeric (including temporal) or all strings; mixing
  string and numeric neighbors reports an error.
- Temporal columns must share the exact same dtype, like comparisons.
- Nulls skip the row, mirroring comparison semantics.

## Column sums

```python
# c == a + b in every row
fl.checks.table.sum_columns(["a", "b"], "c")
# tolerate floating-point slack (defaults to exact, tolerance=0.0)
fl.checks.table.sum_columns(["price", "tax"], "total", tolerance=1e-9)
```

- All columns must be numeric; integer columns with no float or mixed
  integer domains are summed exactly (no rounding noise).
- Any float column (or a mix of signed/unsigned integers) switches the row
  to `f64` arithmetic, where `tolerance` applies.
- Null operands skip the row. When generating data from a schema with this
  check, violating target cells are recomputed from the operands (within the
  column's own constraints), falling back to joint re-sampling.

## Implication (presence)

```python
# if 'left' is present, 'by' must be present too
fl.checks.table.implies("left", "by")
```

Rows with a null `left` never fail; rows with a non-null `left` and a null
`by` are reported.

## Functional dependency

```python
# one 'b' value per distinct 'a' value
fl.checks.table.determines("a", "b")
```

The first non-null `b` seen for each `a` is canonical; every row whose `b`
disagrees with it is reported together with the canonical row. Null `a` or
null `b` cells do not participate.

## Temporal gap

```python
from datetime import timedelta

# shipping_date <= delivery_date, at most 14 days apart
fl.checks.table.within("shipping_date", "delivery_date", timedelta(days=14))
```

- Both columns must be temporal and share the exact same dtype; anything
  else reports an error.
- The gap is measured in the dtype's natural unit (days for `Date`,
  nanoseconds for `Datetime`/`Time`), and `max_gap` is inclusive.
- Nulls skip the row.

## Errors and serialization

Table-check errors carry all involved columns:

```python
report = schema.validate(df, raise_on_error=False)
err = report.errors[0]
err.columns  # e.g. ["start", "end"]
err.column   # primary column ("start") — backward-compatible alias
err.rows     # failing row indices
```

Table checks persist through `to_json()` / `from_json()` like everything
else (see [Serialization](serialization.md)).

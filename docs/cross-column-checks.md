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
        fl.lt_column("start", "end"),
        fl.unique_together(["country", "city"]),
    ],
)
```

## Comparisons

| Function | Rule |
|---|---|
| `fl.lt_column(a, b)` | `a < b` |
| `fl.lte_column(a, b)` | `a <= b` |
| `fl.gt_column(a, b)` | `a > b` |
| `fl.gte_column(a, b)` | `a >= b` |
| `fl.eq_columns(a, b)` | `a == b` |
| `fl.ne_columns(a, b)` | `a != b` |

Semantics:

- Both columns must be numeric (including temporal) or both strings;
  mixing kinds reports an error.
- Rows where either side is null are **skipped** — enforce presence with
  `nullable=False` on the columns if nulls must fail.
- The two column names must differ; otherwise construction raises
  `ValueError`.
- A missing column reports a `ColumnExists` error instead of row failures.

Each failure is reported once with `columns == [a, b]` and the failing row
indices.

## Uniqueness across columns

```python
fl.unique_together(["country", "city"])
```

Fails when the same combination appears in more than one row; all rows in
each duplicated group are reported. Needs at least two columns.

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

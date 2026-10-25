# Table checks

Every function in `framelaw.checks.table` returns a `DataFrameCheck` value.
Pass them to `Schema(..., dataframe_checks=[...])`:

```python
fl.Schema(
    {"start": fl.Column(pl.Date), "end": fl.Column(pl.Date)},
    dataframe_checks=[fl.checks.table.lt_column("start", "end")],
)
```

See the [Cross-column checks guide](../cross-column-checks.md) for semantics,
null handling, and serialization.

::: framelaw.DataFrameCheck

::: framelaw.checks.table
    options:
      members:
        - gt_column
        - lt_column
        - gte_column
        - lte_column
        - eq_columns
        - ne_columns
        - ordered
        - sum_columns
        - unique_together
        - implies
        - determines
        - within

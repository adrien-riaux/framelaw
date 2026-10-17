# Error reports

Validation never returns bare booleans. Every failure is structured data:
which columns, which check, which rows, and a human-readable message.

## The report object

```python
report = schema.validate(df, raise_on_error=False)
report.is_valid()       # bool
report.summary()        # multi-line human-readable string
report.errors           # list of ValidationError
report.to_dataframe()   # Polars DataFrame, one row per failing cell
report.raise_if_invalid()  # re-raise as ValueError, or None if valid
```

`validate()` returns `None` when the frame is valid — so `report is None`
also means "valid" for the `validate`/`validate_lazy` calls.

Example summary:

```text
Column 'age': rows [1], check 'GreaterThanF64': Values in 'age' must be > 0
Columns 'start', 'end': rows [0], check 'ColumnComparison': Condition 'start < end' failed for 1 row(s) at [0]
```

## The error object

Each `ValidationError` exposes:

| Attribute | Meaning |
|---|---|
| `column` | primary column (first of `columns`; kept for compatibility) |
| `columns` | all involved columns — one for column checks, several for table checks |
| `check` | check name, e.g. `GreaterThanF64`, `MatchesRegex`, `ColumnComparison`, `Dtype`, `ColumnExists` |
| `rows` | failing row indices (empty for schema-level errors like missing columns or dtype mismatches) |
| `message` | human-readable description |

Special check names you may see that are not user-constructed:

- `Dtype` — column dtype does not match the schema.
- `ColumnExists` — a required column is missing from the frame.
- `NotNull` — emitted by `nullable=False` (as well as `fl.not_null()`).

## Report as a DataFrame

`to_dataframe()` expands every failing row into its own row, which makes it
easy to join failures back onto the original data:

```python
failures = report.to_dataframe()
# columns: ["column", "columns", "check", "row", "message"]
```

- `column`: primary column; `columns`: `+`-joined list of all columns.
- `row`: `null` for schema-level errors, otherwise the failing row index.

# Error reports

Validation never returns bare booleans. Every failure is structured data:
which columns, which check, which rows, and a human-readable message.

## The report object

```python
import polars as pl
import framelaw as fl

schema = fl.Schema(
    {
        "age": fl.Column(pl.Int64, checks=[fl.checks.column.gt(0)]),
        "email": fl.Column(pl.String, checks=[fl.checks.column.matches(r".+@.+\..+")]),
    }
)
df = pl.DataFrame({"age": [30, -1], "email": ["a@example.com", "nope"]})

report = schema.validate(df, raise_on_error=False)
report.is_valid()       # bool
report.summary()        # multi-line human-readable string
report.errors           # list of ValidationError
report.to_dataframe()   # Polars DataFrame, one row per failing cell
# report.raise_if_invalid()  # raises fl.SchemaError with the summary
```

`validate()` returns `None` when the frame is valid — so `report is None`
also means "valid" for the `validate`/`validate_lazy` calls.

Example summary:

```text
Validation failed with 2 errors
- Column 'age' (rows 1): value must be > 0 [gt]
- Columns 'start', 'end' (rows 0): condition 'start < end' failed for 1 row(s) [lt_column]
```

Row lists collapse past five indices (`rows 0, 1, 2, 3, 4 +12 more`); the
full index list stays available on each error's `rows` attribute.
Schema-level errors render `(schema)` instead of a row list.

## The error object

Each `ValidationError` exposes:

| Attribute | Meaning |
|---|---|
| `column` | primary column (first of `columns`; kept for compatibility) |
| `columns` | all involved columns — one for column checks, several for table checks |
| `check` | public check name, e.g. `gt`, `one_of`, `lt_column`, `dtype`, `column_exists` |
| `rows` | failing row indices (empty for schema-level errors like missing columns or dtype mismatches) |
| `message` | human-readable description |

Special check names you may see that are not user-constructed:

- `dtype` — column dtype does not match the schema.
- `column_exists` — a required column is missing from the frame.
- `not_null` — emitted by `nullable=False` (as well as `fl.checks.column.not_null()`).

## Report as a DataFrame

`to_dataframe()` expands every failing row into its own row, which makes it
easy to join failures back onto the original data:

```python
failures = report.to_dataframe()
# columns: ["column", "columns", "check", "row", "message"]
```

- `column`: primary column; `columns`: `+`-joined list of all columns.
- `row`: `null` for schema-level errors, otherwise the failing row index.

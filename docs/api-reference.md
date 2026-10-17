# API reference

Complete reference for the `framelaw` package. Everything below is importable
from the top level (`import framelaw as fl`).

## Classes

### `Schema`

```python
Schema(columns, dataframe_checks=None, lazy=False)
```

| Member | Kind | Description |
|---|---|---|
| `validate(df, raise_on_error=None)` | method | `None` if valid; raises `ValueError` if invalid and raising; else `ValidationReport`. `None` resolves to `not self.lazy`. |
| `validate_lazy(df)` | method | Never raises: `None` if valid, else `ValidationReport`. |
| `sample(n=10, seed=None)` | method | Generate a valid `polars.DataFrame` (see [Data generation](data-generation.md)). |
| `to_json()` | method | Serialize to a JSON string. |
| `from_json(payload)` | staticmethod | Deserialize; accepts v1 and v2 payloads. |
| `from_polars_schema(mapping)` | staticmethod | Build from `{name: polars_dtype}` with no checks. |
| `from_pydantic(model)` | classmethod | Derive from a Pydantic v2 `BaseModel`. |
| `schema[name]` | lookup | Column reference with `add_check(check)`; `KeyError` if missing. |
| `lazy` | property | The schema's lazy flag. |

### `Column`

```python
Column(dtype, nullable=True, checks=None)
```

| Attribute | Description |
|---|---|
| `dtype` | Polars data type the column must match. |
| `nullable` | Whether nulls are allowed (`False` reports offending rows). |
| `checks` | List of check objects. |

### `Check`

Opaque value returned by the check constructors below. Combine freely in a
column's `checks` list.

Single-column constructors: `not_null`, `unique_values` (alias `unique`),
`gt`, `lt`, `gte`, `lte`, `between`, `matches_regex` (alias `matches`),
`is_in` (alias `one_of`), `list_len_between`, `list_not_empty`,
`struct_field(field, checks)`.

### `DataFrameCheck`

Opaque value returned by the table-check constructors. Pass via
`Schema(..., dataframe_checks=[...])`.

Table-check constructors: `gt_column`, `lt_column`, `gte_column`,
`lte_column`, `eq_columns`, `ne_columns`, `unique_together`.

### `ValidationReport`

| Member | Description |
|---|---|
| `errors` | List of `ValidationError`. |
| `is_valid()` | `True` when there are no errors. |
| `summary()` | Human-readable multi-line string. |
| `to_dataframe()` | Polars frame with columns `column`, `columns`, `check`, `row`, `message`. |
| `raise_if_invalid()` | Raise `ValueError` with the summary, or return `None`. |

### `ValidationError`

| Attribute | Description |
|---|---|
| `column` | Primary column (first of `columns`). |
| `columns` | All involved columns. |
| `check` | Check name (`GreaterThanF64`, `ColumnComparison`, `Dtype`, …). |
| `rows` | Failing row indices (`[]` for schema-level errors). |
| `message` | Human-readable description. |

## Functions

### Validation

| Function | Description |
|---|---|
| `infer_schema(df)` | Draft a `Schema` from a DataFrame or LazyFrame. |

### Sampling

| Function | Description |
|---|---|
| `sample(schema, n=10, seed=None)` | Generate a valid frame; equivalent to `schema.sample(n, seed)`. |

### Check constructors

| Function | Signature |
|---|---|
| `not_null()` | — |
| `unique_values()` / `unique()` | — |
| `gt(value)` / `lt(value)` / `gte(value)` / `lte(value)` | `float` threshold |
| `between(min, max)` | inclusive `float` bounds |
| `matches_regex(pattern)` / `matches(pattern)` | regex `str` (validated at construction) |
| `is_in(allowed)` / `one_of(allowed)` | `list[str]` |
| `list_len_between(min_len, max_len)` | `int` bounds, `min_len <= max_len` |
| `list_not_empty()` | — |
| `struct_field(field, checks)` | field name + list of checks |

### Table-check constructors

| Function | Signature |
|---|---|
| `gt_column(left, right)` / `lt_column(left, right)` | column names, must differ |
| `gte_column(left, right)` / `lte_column(left, right)` | column names, must differ |
| `eq_columns(left, right)` / `ne_columns(left, right)` | column names, must differ |
| `unique_together(columns)` | ≥ 2 column names |

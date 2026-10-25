# API reference

Complete reference for the `framelaw` package. Core classes and helpers are
importable from the top level (`import framelaw as fl`); check constructors
live in the `framelaw.checks` submodules.

## Module

### `framelaw.__version__`

`str` — the installed distribution version in PEP 440 form (for example
`"0.1.0"` or `"0.1.1rc1"`), read from the package metadata.

### `framelaw.checks`

Namespace for check constructors: [`framelaw.checks.column`](#check-constructors)
for single-column checks and [`framelaw.checks.table`](#table-check-constructors)
for cross-column checks. The pre-0.3 top-level names (`framelaw.gt`,
`framelaw.lt_column`, ...) still work but emit `DeprecationWarning`.

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

Single-column constructors in `framelaw.checks.column`: `not_null`,
`unique_values` (alias `unique`), `gt`, `lt`, `gte`, `lte`, `between`,
`matches_regex` (alias `matches`), `is_in` (alias `one_of`),
`list_len_between`, `list_not_empty`, `struct_field(field, checks)`.

### `DataFrameCheck`

Opaque value returned by the table-check constructors. Pass via
`Schema(..., dataframe_checks=[...])`.

Table-check constructors in `framelaw.checks.table`: `gt_column`, `lt_column`,
`gte_column`, `lte_column`, `eq_columns`, `ne_columns`, `unique_together`,
`ordered`, `sum_columns`, `implies`, `determines`, `within`.

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

All single-column constructors live in `framelaw.checks.column`.

| Function | Signature |
|---|---|
| `not_null()` | — |
| `unique_values()` / `unique()` | — |
| `gt(value)` / `lt(value)` / `gte(value)` / `lte(value)` | `float` threshold |
| `between(min, max)` | inclusive `float` bounds |
| `matches_regex(pattern)` / `matches(pattern)` | regex `str` (validated at construction) |
| `str_length_between(min_len, max_len)` | inclusive `int` bounds, `min_len <= max_len` |
| `not_blank()` | — |
| `starts_with(prefix)` / `ends_with(suffix)` | `str` prefix / suffix |
| `is_in(allowed)` / `one_of(allowed)` | `list[str]` (string membership) or `list[int | float | bool]` (numeric membership) |
| `list_len_between(min_len, max_len)` | `int` bounds, `min_len <= max_len` |
| `list_not_empty()` | — |
| `struct_field(field, checks)` | field name + list of checks |

### Table-check constructors

All cross-column constructors live in `framelaw.checks.table`.

| Function | Signature |
|---|---|
| `gt_column(left, right)` / `lt_column(left, right)` | column names, must differ |
| `gte_column(left, right)` / `lte_column(left, right)` | column names, must differ |
| `eq_columns(left, right)` / `ne_columns(left, right)` | column names, must differ |
| `ordered(columns, strict)` | ≥ 2 distinct column names |
| `sum_columns(addends, target, tolerance=None)` | ≥ 1 addend, `tolerance >= 0` |
| `implies(left, by)` / `determines(a, b)` | column names, must differ |
| `within(left, right, max_gap)` | column names, must differ; `max_gap` is a `timedelta` |

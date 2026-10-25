# Checks

Single-column checks validate the values of one column and report the failing
row indices. Every check is a plain value constructed by a function in
`framelaw.checks.column`:

```python
fl.Column(pl.Int64, checks=[fl.checks.column.gt(0), fl.checks.column.lt(120)])
```

## Nullability

| Function | Fails when |
|---|---|
| `fl.checks.column.not_null()` | any value is null |

Note: `nullable=False` on the column already enforces non-null and reports
the rows. Adding `fl.checks.column.not_null()` explicitly is equivalent and is what
`from_pydantic` and JSON round-trips produce.

## Uniqueness

| Function (alias) | Fails when |
|---|---|
| `fl.checks.column.unique_values()` (`fl.checks.column.unique()`) | any duplicated value exists |

All rows sharing a duplicated value are reported.

## Numeric comparisons

| Function | Fails when |
|---|---|
| `fl.checks.column.gt(value)` | any value `<= value` |
| `fl.checks.column.lt(value)` | any value `>= value` |
| `fl.checks.column.gte(value)` | any value `< value` |
| `fl.checks.column.lte(value)` | any value `> value` |
| `fl.checks.column.between(min, max)` | any value outside `[min, max]` (inclusive) |

These apply to all integer and float dtypes, plus temporal columns (`Date`,
`Datetime`, `Duration`, `Time`), which compare on their chronological order.
Applying them to a non-numeric column produces a type-mismatch error instead
of row failures.

```python
fl.Column(pl.Float64, checks=[fl.checks.column.between(0.0, 1.0)])
fl.Column(pl.Date, checks=[fl.checks.column.gt(0)])  # dates after the epoch
```

## String checks

| Function (alias) | Fails when |
|---|---|
| `fl.checks.column.matches(pattern)` (`fl.checks.column.matches_regex(pattern)`) | a non-null string does not match the regex |
| `fl.checks.column.one_of(values)` (`fl.checks.column.is_in(values)`) | a non-null string is not in the allowed set |
| `fl.checks.column.str_length_between(min_len, max_len)` | a non-null string length is outside `[min_len, max_len]` characters |
| `fl.checks.column.not_blank()` | a non-null string is empty or whitespace-only |
| `fl.checks.column.starts_with(prefix)` | a non-null string does not start with the prefix |
| `fl.checks.column.ends_with(suffix)` | a non-null string does not end with the suffix |

Nulls are skipped by all string checks — nullability is governed by
`nullable` / `not_null` only. An invalid regex raises `ValueError`
immediately at construction time. Length is counted in characters, not
bytes.

```python
fl.Column(pl.String, checks=[fl.checks.column.matches(r"^[A-Z]{2}$"), fl.checks.column.one_of(["FR", "US"])])
fl.Column(pl.String, checks=[fl.checks.column.not_blank(), fl.checks.column.str_length_between(1, 64)])
fl.Column(pl.String, checks=[fl.checks.column.starts_with("prj_"), fl.checks.column.ends_with("-v1")])
```

## Membership checks

`fl.checks.column.one_of(values)` / `fl.checks.column.is_in(values)` accepts a list of strings, numbers,
or booleans. All elements must share one class — a mixed list raises
`ValueError`:

```python
fl.Column(pl.String, checks=[fl.checks.column.one_of(["FR", "US"])])
fl.Column(pl.Int64, checks=[fl.checks.column.is_in([1, 2, 3])])
fl.Column(pl.Float64, checks=[fl.checks.column.is_in([0.0, 0.5, 1.0])])
fl.Column(pl.Boolean, checks=[fl.checks.column.is_in([True])])
```

Numeric membership compares at `f64` granularity against float columns and
exactly against integer columns (values are widened to `i128`/`u128`
before comparison, so integers beyond the 2^53 float range are never
merged by rounding). Boolean columns are compared as `true`/`false`
through the same integer machinery, so `is_in([True])` only accepts
`True` values.

## Nested checks

| Function | Applies to | Fails when |
|---|---|---|
| `fl.checks.column.list_len_between(min_len, max_len)` | `List`, `Array` | a non-null list length is outside `[min_len, max_len]` |
| `fl.checks.column.list_not_empty()` | `List`, `Array` | a non-null list is empty |
| `fl.checks.column.struct_field(field, checks)` | `Struct` | the subfield fails any of the given checks |

```python
fl.Column(pl.List(pl.Int64), checks=[fl.checks.column.list_len_between(1, 3)])
fl.Column(
    pl.Struct({"city": pl.String, "zip": pl.String}),
    checks=[fl.checks.column.struct_field("zip", [fl.checks.column.matches(r"^\d{5}$")])],
)
```

A missing struct field reports an error with no rows. See
[Nested types](nested-types.md) for details.

## Combining checks

Checks in one column are independent and all run; every failure is reported:

```python
fl.Column(
    pl.Int64,
    nullable=False,
    checks=[fl.checks.column.not_null(), fl.checks.column.unique(), fl.checks.column.gt(0), fl.checks.column.lt(100)],
)
```

Validation itself is parallelized across columns with Rayon, so wide frames
scale with available cores.

# Checks

Single-column checks validate the values of one column and report the failing
row indices. Every check is a plain value constructed by a module-level
function:

```python
fl.Column(pl.Int64, checks=[fl.gt(0), fl.lt(120)])
```

## Nullability

| Function | Fails when |
|---|---|
| `fl.not_null()` | any value is null |

Note: `nullable=False` on the column already enforces non-null and reports
the rows. Adding `fl.not_null()` explicitly is equivalent and is what
`from_pydantic` and JSON round-trips produce.

## Uniqueness

| Function (alias) | Fails when |
|---|---|
| `fl.unique_values()` (`fl.unique()`) | any duplicated value exists |

All rows sharing a duplicated value are reported.

## Numeric comparisons

| Function | Fails when |
|---|---|
| `fl.gt(value)` | any value `<= value` |
| `fl.lt(value)` | any value `>= value` |
| `fl.gte(value)` | any value `< value` |
| `fl.lte(value)` | any value `> value` |
| `fl.between(min, max)` | any value outside `[min, max]` (inclusive) |

These apply to all integer and float dtypes, plus temporal columns (`Date`,
`Datetime`, `Duration`, `Time`), which compare on their chronological order.
Applying them to a non-numeric column produces a type-mismatch error instead
of row failures.

```python
fl.Column(pl.Float64, checks=[fl.between(0.0, 1.0)])
fl.Column(pl.Date, checks=[fl.gt(0)])  # dates after the epoch
```

## String checks

| Function (alias) | Fails when |
|---|---|
| `fl.matches(pattern)` (`fl.matches_regex(pattern)`) | a non-null string does not match the regex |
| `fl.one_of(values)` (`fl.is_in(values)`) | a non-null string is not in the allowed set |

Nulls are skipped by both — nullability is governed by `nullable` / `not_null`
only. An invalid regex raises `ValueError` immediately at construction time.

```python
fl.Column(pl.String, checks=[fl.matches(r"^[A-Z]{2}$"), fl.one_of(["FR", "US"])])
```

## Nested checks

| Function | Applies to | Fails when |
|---|---|---|
| `fl.list_len_between(min_len, max_len)` | `List`, `Array` | a non-null list length is outside `[min_len, max_len]` |
| `fl.list_not_empty()` | `List`, `Array` | a non-null list is empty |
| `fl.struct_field(field, checks)` | `Struct` | the subfield fails any of the given checks |

```python
fl.Column(pl.List(pl.Int64), checks=[fl.list_len_between(1, 3)])
fl.Column(
    pl.Struct({"city": pl.String, "zip": pl.String}),
    checks=[fl.struct_field("zip", [fl.matches(r"^\d{5}$")])],
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
    checks=[fl.not_null(), fl.unique(), fl.gt(0), fl.lt(100)],
)
```

Validation itself is parallelized across columns with Rayon, so wide frames
scale with available cores.

# Single-column checks

Every function in `framelaw.checks.column` returns a `Check` value. Attach
checks to a column with `checks=[...]`:

```python
fl.Column(pl.Int64, checks=[fl.checks.column.gt(0), fl.checks.column.lt(120)])
```

Value checks skip nulls; nullability is governed by `nullable` or
`fl.checks.column.not_null()`. See the [Checks guide](../checks.md) for
examples and edge cases.

::: framelaw.Check

::: framelaw.checks.column
    options:
      members:
        - not_null
        - unique_values
        - unique
        - gt
        - lt
        - gte
        - lte
        - between
        - matches
        - matches_regex
        - one_of
        - is_in
        - str_length_between
        - not_blank
        - starts_with
        - ends_with
        - list_len_between
        - list_not_empty
        - struct_field

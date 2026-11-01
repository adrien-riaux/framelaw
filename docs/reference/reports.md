# Reports

`Schema.validate(..., raise_on_error=False)`, `validate_lazy()`, and chunked
validation return a `ValidationReport`. A report holds a list of
`ValidationError` values and can render itself as a Polars frame. When
raising is enabled, failures surface as `SchemaError`.

::: framelaw.ValidationReport

::: framelaw.ValidationError

::: framelaw.SchemaError

## See also

- [Error reports guide](../error-reports.md) — report shape and examples.
- [Lazy validation](../lazy-validation.md) — reports instead of exceptions.

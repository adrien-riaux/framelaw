"""Framelaw: A blazingly fast Polars schema validation engine written in Rust.

This module provides the main entrypoint for defining schemas
and validating Polars DataFrames using high-performance Rust logic.

Check constructors live in the :mod:`framelaw.checks` package:
:mod:`framelaw.checks.column` for single-column checks and
:mod:`framelaw.checks.table` for cross-column checks.
"""

from importlib.metadata import PackageNotFoundError, version

from . import checks
from ._compat import (
    between,
    determines,
    ends_with,
    eq_columns,
    gt,
    gt_column,
    gte,
    gte_column,
    implies,
    is_in,
    list_len_between,
    list_not_empty,
    lt,
    lt_column,
    lte,
    lte_column,
    matches,
    matches_regex,
    ne_columns,
    not_blank,
    not_null,
    one_of,
    ordered,
    starts_with,
    str_length_between,
    struct_field,
    sum_columns,
    unique,
    unique_together,
    unique_values,
    within,
)
from ._framelaw import (
    Check,
    Column,
    DataFrameCheck,
    Schema,
    ValidationError,
    ValidationReport,
    infer_schema,
    sample,
)

try:
    __version__ = version("framelaw")
except PackageNotFoundError:
    __version__ = "0.0.0+unknown"

__all__ = [
    "__version__",
    "Check",
    "Column",
    "DataFrameCheck",
    "Schema",
    "ValidationError",
    "ValidationReport",
    "checks",
    "infer_schema",
    "sample",
    # Deprecated top-level check aliases, kept for compatibility.
    "not_null",
    "unique_values",
    "gt",
    "gte",
    "lt",
    "lte",
    "between",
    "matches",
    "matches_regex",
    "one_of",
    "is_in",
    "unique",
    "list_len_between",
    "list_not_empty",
    "struct_field",
    "str_length_between",
    "not_blank",
    "starts_with",
    "ends_with",
    "gt_column",
    "lt_column",
    "gte_column",
    "lte_column",
    "eq_columns",
    "ne_columns",
    "unique_together",
    "ordered",
    "sum_columns",
    "implies",
    "determines",
    "within",
]

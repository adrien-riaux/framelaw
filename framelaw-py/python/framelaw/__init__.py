"""Framelaw: A blazingly fast Polars schema validation engine written in Rust.

This module provides the main entrypoint for defining schemas
and validating Polars DataFrames using high-performance Rust logic.
"""

from ._framelaw import (
    Check,
    Column,
    Schema,
    between,
    gt,
    is_in,
    lt,
    matches,
    matches_regex,
    not_null,
    one_of,
    unique,
    unique_values,
)

__all__ = [
    "Check",
    "Column",
    "Schema",
    "not_null",
    "unique_values",
    "gt",
    "lt",
    "between",
    "matches",
    "matches_regex",
    "one_of",
    "is_in",
    "unique",
]

"""Framelaw: A blazingly fast Polars schema validation engine written in Rust.

This module provides the main entrypoint for defining schemas
and validating Polars DataFrames using high-performance Rust logic.
"""

from framelaw.framelaw import (
    Check,
    Column,
    Schema,
    between,
    gt,
    is_in,
    lt,
    matches_regex,
    not_null,
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
    "matches_regex",
    "is_in",
]

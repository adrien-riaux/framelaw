"""Framelaw: A blazingly fast Polars schema validation engine written in Rust.

This module provides the main entrypoint for defining schemas
and validating Polars DataFrames using high-performance Rust logic.

Check constructors live in the :mod:`framelaw.checks` package:
:mod:`framelaw.checks.column` for single-column checks and
:mod:`framelaw.checks.table` for cross-column checks.
"""

from importlib.metadata import PackageNotFoundError, version

from . import checks
from ._decorator import guard
from ._framelaw import (
    Check,
    Column,
    ColumnRef,
    DataFrameCheck,
    Schema,
    SchemaError,
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
    "ColumnRef",
    "DataFrameCheck",
    "Schema",
    "SchemaError",
    "ValidationError",
    "ValidationReport",
    "checks",
    "guard",
    "infer_schema",
    "sample",
]

"""Deprecated top-level aliases for the check constructors.

Before ``framelaw.checks`` existed, every check was exposed directly under
``framelaw``. These wrappers keep the old import paths working while pointing
users at the scoped namespaces.
"""

from __future__ import annotations

import warnings
from datetime import timedelta

from ._framelaw import Check, DataFrameCheck
from .checks import column, table


def _warn(name: str, replacement: str) -> None:
    """Emit the migration warning for a legacy top-level check alias."""
    warnings.warn(
        f"framelaw.{name} is deprecated; use framelaw.{replacement} instead.",
        DeprecationWarning,
        stacklevel=3,
    )


def not_null() -> Check:
    """Deprecated alias of :func:`framelaw.checks.column.not_null`."""
    _warn("not_null", "checks.column.not_null")
    return column.not_null()


def unique_values() -> Check:
    """Deprecated alias of :func:`framelaw.checks.column.unique_values`."""
    _warn("unique_values", "checks.column.unique_values")
    return column.unique_values()


def unique() -> Check:
    """Deprecated alias of :func:`framelaw.checks.column.unique`."""
    _warn("unique", "checks.column.unique")
    return column.unique()


def gt(value: float) -> Check:
    """Deprecated alias of :func:`framelaw.checks.column.gt`."""
    _warn("gt", "checks.column.gt")
    return column.gt(value)


def lt(value: float) -> Check:
    """Deprecated alias of :func:`framelaw.checks.column.lt`."""
    _warn("lt", "checks.column.lt")
    return column.lt(value)


def gte(value: float) -> Check:
    """Deprecated alias of :func:`framelaw.checks.column.gte`."""
    _warn("gte", "checks.column.gte")
    return column.gte(value)


def lte(value: float) -> Check:
    """Deprecated alias of :func:`framelaw.checks.column.lte`."""
    _warn("lte", "checks.column.lte")
    return column.lte(value)


def between(min: float, max: float) -> Check:
    """Deprecated alias of :func:`framelaw.checks.column.between`."""
    _warn("between", "checks.column.between")
    return column.between(min, max)


def matches(pattern: str) -> Check:
    """Deprecated alias of :func:`framelaw.checks.column.matches`."""
    _warn("matches", "checks.column.matches")
    return column.matches(pattern)


def matches_regex(pattern: str) -> Check:
    """Deprecated alias of :func:`framelaw.checks.column.matches_regex`."""
    _warn("matches_regex", "checks.column.matches_regex")
    return column.matches_regex(pattern)


def is_in(allowed: list[str | int | float | bool]) -> Check:
    """Deprecated alias of :func:`framelaw.checks.column.is_in`."""
    _warn("is_in", "checks.column.is_in")
    return column.is_in(allowed)


def one_of(allowed: list[str | int | float | bool]) -> Check:
    """Deprecated alias of :func:`framelaw.checks.column.one_of`."""
    _warn("one_of", "checks.column.one_of")
    return column.one_of(allowed)


def str_length_between(min_len: int, max_len: int) -> Check:
    """Deprecated alias of :func:`framelaw.checks.column.str_length_between`."""
    _warn("str_length_between", "checks.column.str_length_between")
    return column.str_length_between(min_len, max_len)


def not_blank() -> Check:
    """Deprecated alias of :func:`framelaw.checks.column.not_blank`."""
    _warn("not_blank", "checks.column.not_blank")
    return column.not_blank()


def starts_with(prefix: str) -> Check:
    """Deprecated alias of :func:`framelaw.checks.column.starts_with`."""
    _warn("starts_with", "checks.column.starts_with")
    return column.starts_with(prefix)


def ends_with(suffix: str) -> Check:
    """Deprecated alias of :func:`framelaw.checks.column.ends_with`."""
    _warn("ends_with", "checks.column.ends_with")
    return column.ends_with(suffix)


def list_len_between(min_len: int, max_len: int) -> Check:
    """Deprecated alias of :func:`framelaw.checks.column.list_len_between`."""
    _warn("list_len_between", "checks.column.list_len_between")
    return column.list_len_between(min_len, max_len)


def list_not_empty() -> Check:
    """Deprecated alias of :func:`framelaw.checks.column.list_not_empty`."""
    _warn("list_not_empty", "checks.column.list_not_empty")
    return column.list_not_empty()


def struct_field(field: str, checks: list[Check]) -> Check:
    """Deprecated alias of :func:`framelaw.checks.column.struct_field`."""
    _warn("struct_field", "checks.column.struct_field")
    return column.struct_field(field, checks)


def gt_column(left: str, right: str) -> DataFrameCheck:
    """Deprecated alias of :func:`framelaw.checks.table.gt_column`."""
    _warn("gt_column", "checks.table.gt_column")
    return table.gt_column(left, right)


def lt_column(left: str, right: str) -> DataFrameCheck:
    """Deprecated alias of :func:`framelaw.checks.table.lt_column`."""
    _warn("lt_column", "checks.table.lt_column")
    return table.lt_column(left, right)


def gte_column(left: str, right: str) -> DataFrameCheck:
    """Deprecated alias of :func:`framelaw.checks.table.gte_column`."""
    _warn("gte_column", "checks.table.gte_column")
    return table.gte_column(left, right)


def lte_column(left: str, right: str) -> DataFrameCheck:
    """Deprecated alias of :func:`framelaw.checks.table.lte_column`."""
    _warn("lte_column", "checks.table.lte_column")
    return table.lte_column(left, right)


def eq_columns(left: str, right: str) -> DataFrameCheck:
    """Deprecated alias of :func:`framelaw.checks.table.eq_columns`."""
    _warn("eq_columns", "checks.table.eq_columns")
    return table.eq_columns(left, right)


def ne_columns(left: str, right: str) -> DataFrameCheck:
    """Deprecated alias of :func:`framelaw.checks.table.ne_columns`."""
    _warn("ne_columns", "checks.table.ne_columns")
    return table.ne_columns(left, right)


def unique_together(columns: list[str]) -> DataFrameCheck:
    """Deprecated alias of :func:`framelaw.checks.table.unique_together`."""
    _warn("unique_together", "checks.table.unique_together")
    return table.unique_together(columns)


def ordered(columns: list[str], strict: bool) -> DataFrameCheck:
    """Deprecated alias of :func:`framelaw.checks.table.ordered`."""
    _warn("ordered", "checks.table.ordered")
    return table.ordered(columns, strict)


def sum_columns(addends: list[str], target: str, tolerance: float | None = None) -> DataFrameCheck:
    """Deprecated alias of :func:`framelaw.checks.table.sum_columns`."""
    _warn("sum_columns", "checks.table.sum_columns")
    return table.sum_columns(addends, target, tolerance)


def implies(left: str, by: str) -> DataFrameCheck:
    """Deprecated alias of :func:`framelaw.checks.table.implies`."""
    _warn("implies", "checks.table.implies")
    return table.implies(left, by)


def determines(a: str, b: str) -> DataFrameCheck:
    """Deprecated alias of :func:`framelaw.checks.table.determines`."""
    _warn("determines", "checks.table.determines")
    return table.determines(a, b)


def within(left: str, right: str, max_gap: timedelta) -> DataFrameCheck:
    """Deprecated alias of :func:`framelaw.checks.table.within`."""
    _warn("within", "checks.table.within")
    return table.within(left, right, max_gap)

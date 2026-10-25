"""Check constructors grouped by validation scope.

Use :mod:`framelaw.checks.column` for single-column checks that return a
:class:`framelaw.Check`, and :mod:`framelaw.checks.table` for cross-column
checks that return a :class:`framelaw.DataFrameCheck`.

Examples:
    >>> import framelaw as fl
    >>> fl.checks.column.gt(18)  # doctest: +SKIP
    >>> fl.checks.table.unique_together(["order_id", "line_no"])  # doctest: +SKIP
"""

from . import column, table

__all__ = ["column", "table"]

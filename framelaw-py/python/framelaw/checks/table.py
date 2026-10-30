"""Cross-column (table) check constructors.

Every function in this module returns a :class:`framelaw.DataFrameCheck` that
can be attached to a :class:`framelaw.Schema` through ``dataframe_checks=[...]``.
"""

from .._framelaw import (
    check_expr,
    check_mask,
    check_rows,
    determines,
    eq_columns,
    gt_column,
    gte_column,
    implies,
    lt_column,
    lte_column,
    ne_columns,
    ordered,
    sum_columns,
    unique_together,
    within,
)

__all__ = [
    "check_expr",
    "check_mask",
    "check_rows",
    "determines",
    "eq_columns",
    "gt_column",
    "gte_column",
    "implies",
    "lt_column",
    "lte_column",
    "ne_columns",
    "ordered",
    "sum_columns",
    "unique_together",
    "within",
]

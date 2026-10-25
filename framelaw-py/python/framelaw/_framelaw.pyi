"""Type stubs for FrameLaw's compiled Rust extension.

This file is never executed at runtime (the import system ignores `.pyi`
files when a compiled module is present). It exists so type checkers and
IDEs understand the native API. Keep signatures in sync with
`framelaw-py/src/lib.rs`.
"""

from datetime import timedelta
from typing import Any

class Check:
    """Opaque single-column check.

    Checks are created by the constructors in :mod:`framelaw.checks.column`
    and attached to a :class:`Column` through ``checks=[...]``. They are plain
    values and can be reused across columns and schemas.
    """

class DataFrameCheck:
    """Opaque cross-column (table) check.

    Table checks are created by the constructors in
    :mod:`framelaw.checks.table` and passed to
    ``Schema(..., dataframe_checks=[...])``.
    """

class Column:
    """Specification of one column in a :class:`Schema`.

    Args:
        dtype: Polars data type the column must match exactly (structurally
            for nested types). Examples: ``pl.Int64``, ``pl.String``,
            ``pl.Date``, ``pl.List(pl.Int64)``,
            ``pl.Struct({"zip": pl.String})``.
        nullable: Whether null values are allowed. When ``False``, nulls are
            reported as a ``NotNull`` error with the offending row indices.
        checks: Single-column checks to run, usually built with
            :mod:`framelaw.checks.column`.

    Attributes:
        dtype: The Polars dtype this column requires.
        nullable: Whether nulls are allowed.
        checks: The checks attached to this column.

    Examples:
        >>> import polars as pl
        >>> import framelaw as fl
        >>> column = fl.Column(
        ...     pl.Int64, nullable=False, checks=[fl.checks.column.gt(0)]
        ... )
    """

    dtype: Any
    nullable: bool
    checks: list[Check]

    def __init__(
        self,
        dtype: Any,
        nullable: bool = True,
        checks: list[Check] | None = None,
    ) -> None: ...

class ColumnRef:
    """Mutable reference to one schema column, returned by ``schema[name]``.

    A column reference is the intended way to tighten a schema after
    inference: look up the column, then append checks.
    """

    def add_check(self, check: Check) -> None:
        """Append a check to this column.

        Args:
            check: Single-column check to add.

        Raises:
            KeyError: If the referenced column no longer exists in the schema.
        """

class Schema:
    """A collection of column specs plus optional cross-column checks.

    A schema is the single object you define, persist, infer, and validate
    with. Column names are matched exactly; missing columns and dtype
    mismatches are reported as schema-level errors.

    Args:
        columns: Mapping from column name to :class:`Column`.
        dataframe_checks: Cross-column checks, usually built with
            :mod:`framelaw.checks.table`.
        lazy: When ``True``, :meth:`validate` returns a
            :class:`ValidationReport` instead of raising. An explicit
            ``raise_on_error`` argument still wins per call.

    Attributes:
        lazy: The schema-wide lazy flag.

    Examples:
        >>> import polars as pl
        >>> import framelaw as fl
        >>> schema = fl.Schema(
        ...     {
        ...         "user_id": fl.Column(
        ...             pl.Int64, nullable=False, checks=[fl.checks.column.unique()]
        ...         ),
        ...         "age": fl.Column(
        ...             pl.Int64, checks=[fl.checks.column.gt(0)]
        ...         ),
        ...     }
        ... )
    """

    def __init__(
        self,
        columns: dict[str, Column],
        dataframe_checks: list[DataFrameCheck] | None = None,
        lazy: bool = False,
    ) -> None: ...
    @property
    def lazy(self) -> bool:
        """Whether this schema returns reports by default instead of raising."""

    def validate(
        self, df: Any, raise_on_error: bool | None = None, chunk_rows: int | None = None
    ) -> ValidationReport | None:
        """Validate a Polars ``DataFrame`` or ``LazyFrame``.

        Args:
            df: The frame to validate. Lazy frames are collected at the
                validation boundary, or sliced and collected per batch when
                ``chunk_rows`` is set.
            raise_on_error: Whether to raise ``ValueError`` on failure.
                ``None`` resolves to ``not self.lazy``.
            chunk_rows: Validate in batches of this many rows. Chunked
                validation always returns a report instead of raising, and
                dataset-wide checks (``unique``, ``unique_together``,
                ``determines``) only see one batch at a time.

        Returns:
            ``None`` when the frame is valid. When invalid, raises
            ``ValueError`` if raising, otherwise returns a
            :class:`ValidationReport`.

        Raises:
            ValueError: If the frame is invalid and raising is enabled, or if
                ``chunk_rows`` is zero.
            TypeError: If ``df`` is neither a Polars ``DataFrame`` nor
                ``LazyFrame``.
        """

    def validate_lazy(self, df: Any) -> ValidationReport | None:
        """Validate without ever raising.

        Args:
            df: Polars ``DataFrame`` or ``LazyFrame`` to validate.

        Returns:
            ``None`` when the frame is valid, otherwise a
            :class:`ValidationReport`.
        """

    def sample(self, n: int = 10, seed: int | None = None) -> Any:
        """Generate a valid Polars ``DataFrame`` from this schema.

        Args:
            n: Number of rows to generate.
            seed: Optional RNG seed for reproducible output.

        Returns:
            A ``polars.DataFrame`` satisfying the schema's checks.

        Raises:
            ValueError: If the schema is contradictory or cannot be
                synthesized (for example an impossible ``unique`` domain).
        """

    def to_json(self) -> str:
        """Serialize the schema to a compact JSON string.

        Returns:
            JSON with the schema version, every column (dtype, nullability,
            checks), every table check, and the ``lazy`` flag.

        Raises:
            ValueError: If a dtype cannot be represented (for example a
                timezone-aware datetime).
        """

    @staticmethod
    def from_polars_schema(schema: dict[str, Any]) -> Schema:
        """Build a schema from a mapping of column names to Polars dtypes.

        The result has no value checks; it is a typed starting point that can
        be extended with ``schema[name].add_check(...)``.

        Args:
            schema: Mapping such as ``{"age": pl.Int64, "email": pl.String}``.

        Returns:
            A schema with the given dtypes and ``nullable=True`` columns.
        """

    @staticmethod
    def from_json(payload: str) -> Schema:
        """Deserialize a schema produced by :meth:`to_json`.

        Args:
            payload: JSON string. Versions 1 and 2 are accepted.

        Returns:
            The restored schema.

        Raises:
            ValueError: If the JSON is malformed, the version is unknown, or
                a dtype tag is unsupported.
        """

    @classmethod
    def from_pydantic(cls, model: Any) -> Schema:
        """Derive a schema from a Pydantic v2 ``BaseModel`` class.

        Field types become Polars dtypes; ``Field`` constraints become checks.
        The model must expose ``__pydantic_core_schema__``.

        Args:
            model: A Pydantic v2 model class.

        Returns:
            A schema with one column per model field.

        Raises:
            TypeError: If a field type is unsupported (for example ``dict``
                or a nested model).
        """

    def __getitem__(self, name: str) -> ColumnRef:
        """Look up a column by name.

        Args:
            name: Column name.

        Returns:
            A mutable :class:`ColumnRef` with :meth:`ColumnRef.add_check`.

        Raises:
            KeyError: If the column is not part of the schema.
        """

class ValidationError:
    """One validation failure.

    Attributes:
        column: Primary column (the first entry of ``columns``). Kept for
            compatibility with single-column errors.
        columns: All columns involved in the failure. One entry for column
            checks, several for table checks.
        check: Check name, for example ``GreaterThanF64``, ``MatchesRegex``,
            ``ColumnComparison``, ``Dtype``, or ``ColumnExists``.
        rows: Failing row indices. Empty for schema-level errors such as a
            missing column or a dtype mismatch.
        message: Human-readable description of the failure.
    """

    column: str
    columns: list[str]
    check: str
    rows: list[int]
    message: str

class ValidationReport:
    """Structured result of a validation that did not raise.

    Attributes:
        errors: Every failure found in the frame.

    Examples:
        >>> report = schema.validate(df, raise_on_error=False)
        >>> if report is not None:
        ...     print(report.summary())
    """

    errors: list[ValidationError]

    def is_valid(self) -> bool:
        """Return ``True`` when the report has no errors."""

    def summary(self) -> str:
        """Render a human-readable multi-line summary of every error."""

    def to_dataframe(self) -> Any:
        """Expand failures into a Polars ``DataFrame``.

        Returns:
            A frame with columns ``column``, ``columns``, ``check``,
            ``row``, and ``message``. Schema-level errors have a null
            ``row``.
        """

    def raise_if_invalid(self) -> None:
        """Raise ``ValueError`` with the summary, or return ``None`` if valid."""

def not_null() -> Check:
    """Reject null values in the column.

    Equivalent to ``nullable=False``, but explicit and serializable. Nulls
    are reported with their row indices.

    Returns:
        A check that fails on any null value.
    """

def unique_values() -> Check:
    """Reject duplicated values.

    All rows sharing a duplicated value are reported, not just the first
    occurrence. Nulls participate in the duplicate scan, so multiple nulls in
    a nullable column are reported as duplicates.

    Returns:
        A check that fails when any non-null value appears more than once.
    """

def unique() -> Check:
    """Alias of :func:`unique_values`."""

def gt(value: float) -> Check:
    """Require every non-null value to be strictly greater than ``value``.

    Applies to numeric and temporal columns (dates and times compare on their
    chronological order). Nulls are skipped.

    Args:
        value: Exclusive lower bound.

    Returns:
        A check that fails when a value is ``<= value``.
    """

def lt(value: float) -> Check:
    """Require every non-null value to be strictly less than ``value``.

    Args:
        value: Exclusive upper bound.

    Returns:
        A check that fails when a value is ``>= value``.
    """

def gte(value: float) -> Check:
    """Require every non-null value to be greater than or equal to ``value``.

    Args:
        value: Inclusive lower bound.

    Returns:
        A check that fails when a value is ``< value``.
    """

def lte(value: float) -> Check:
    """Require every non-null value to be less than or equal to ``value``.

    Args:
        value: Inclusive upper bound.

    Returns:
        A check that fails when a value is ``> value``.
    """

def between(min: float, max: float) -> Check:
    """Require every non-null value to fall inside an inclusive range.

    Args:
        min: Inclusive lower bound.
        max: Inclusive upper bound.

    Returns:
        A check that fails when a value is outside ``[min, max]``.
    """

def matches_regex(pattern: str) -> Check:
    """Require every non-null string to match a regular expression.

    The regex is validated eagerly: an invalid pattern raises
    ``ValueError`` at construction time rather than during validation.

    Args:
        pattern: Rust ``regex`` crate pattern.

    Returns:
        A check that fails on non-matching non-null strings.

    Raises:
        ValueError: If ``pattern`` is not a valid regular expression.
    """

def matches(pattern: str) -> Check:
    """Alias of :func:`matches_regex`."""

def is_in(allowed: list[str | int | float | bool]) -> Check:
    """Require every non-null value to be a member of ``allowed``.

    The list must be non-empty and homogeneous: either all strings, or all
    numbers/booleans. Numeric membership is exact for integer columns and
    compares at ``f64`` granularity for float columns. Boolean values are
    handled as ``0``/``1`` internally, so ``is_in([True])`` accepts only
    ``True``.

    Args:
        allowed: Allowed values. Mixed string/numeric lists are rejected.

    Returns:
        A check that fails when a value is not in ``allowed``.

    Raises:
        ValueError: If the list is empty, mixes strings with numbers, or
            contains a non-finite float.
    """

def one_of(allowed: list[str | int | float | bool]) -> Check:
    """Alias of :func:`is_in`."""

def str_length_between(min_len: int, max_len: int) -> Check:
    """Bound the character length of non-null strings.

    Length is counted in characters, not bytes. Nulls are skipped.

    Args:
        min_len: Inclusive minimum length.
        max_len: Inclusive maximum length.

    Returns:
        A check that fails when a string's length is outside the range.

    Raises:
        ValueError: If ``min_len > max_len``.
    """

def not_blank() -> Check:
    """Reject empty and whitespace-only non-null strings.

    Returns:
        A check that fails when a string is ``""`` or contains only
        whitespace.
    """

def starts_with(prefix: str) -> Check:
    """Require every non-null string to start with ``prefix``.

    Args:
        prefix: Required prefix.

    Returns:
        A check that fails on non-matching non-null strings.
    """

def ends_with(suffix: str) -> Check:
    """Require every non-null string to end with ``suffix``.

    Args:
        suffix: Required suffix.

    Returns:
        A check that fails on non-matching non-null strings.
    """

def list_len_between(min_len: int, max_len: int) -> Check:
    """Bound the length of non-null ``List`` and ``Array`` cells.

    Args:
        min_len: Inclusive minimum length.
        max_len: Inclusive maximum length.

    Returns:
        A check that fails when a list length is outside the range.

    Raises:
        ValueError: If ``min_len > max_len``.
    """

def list_not_empty() -> Check:
    """Reject empty non-null ``List`` and ``Array`` cells.

    Returns:
        A check that fails when a list has length zero.
    """

def struct_field(field: str, checks: list[Check]) -> Check:
    """Apply single-column checks to a subfield of a ``Struct`` column.

    Failures are reported against the parent column, with the field named in
    the message. A missing field reports an error with an empty row list.

    Args:
        field: Struct field name.
        checks: Checks to run against the extracted field values.

    Returns:
        A check that fails when the subfield fails any of ``checks``.
    """

def gt_column(left: str, right: str) -> DataFrameCheck:
    """Require ``left > right`` in every row.

    Both columns must be numeric (including temporal) or both strings.
    Temporal columns must share the exact same dtype. Rows where either side
    is null are skipped.

    Args:
        left: Left column name.
        right: Right column name.

    Returns:
        A table check that reports failing rows.

    Raises:
        ValueError: If ``left == right``.
    """

def lt_column(left: str, right: str) -> DataFrameCheck:
    """Require ``left < right`` in every row.

    Args:
        left: Left column name.
        right: Right column name.

    Returns:
        A table check that reports failing rows.

    Raises:
        ValueError: If ``left == right``.
    """

def gte_column(left: str, right: str) -> DataFrameCheck:
    """Require ``left >= right`` in every row.

    Args:
        left: Left column name.
        right: Right column name.

    Returns:
        A table check that reports failing rows.

    Raises:
        ValueError: If ``left == right``.
    """

def lte_column(left: str, right: str) -> DataFrameCheck:
    """Require ``left <= right`` in every row.

    Args:
        left: Left column name.
        right: Right column name.

    Returns:
        A table check that reports failing rows.

    Raises:
        ValueError: If ``left == right``.
    """

def eq_columns(left: str, right: str) -> DataFrameCheck:
    """Require ``left == right`` in every row.

    Args:
        left: Left column name.
        right: Right column name.

    Returns:
        A table check that reports failing rows.

    Raises:
        ValueError: If ``left == right``.
    """

def ne_columns(left: str, right: str) -> DataFrameCheck:
    """Require ``left != right`` in every row.

    Args:
        left: Left column name.
        right: Right column name.

    Returns:
        A table check that reports failing rows.

    Raises:
        ValueError: If ``left == right``.
    """

def unique_together(columns: list[str]) -> DataFrameCheck:
    """Require each combination of ``columns`` to be unique across rows.

    All rows in a duplicated group are reported. Nulls participate in the
    duplicate scan, so multiple rows with the same null combination are
    reported as duplicates.

    Args:
        columns: At least two column names.

    Returns:
        A table check that fails on duplicate combinations.

    Raises:
        ValueError: If fewer than two columns are provided.
    """

def ordered(columns: list[str], strict: bool) -> DataFrameCheck:
    """Require a per-row ordering chain across ``columns``.

    With ``strict=False``, adjacent values may be equal; with ``strict=True``
    they must be strictly increasing. All columns must be numeric (including
    temporal) or all strings. Nulls skip the row.

    Args:
        columns: At least two distinct, non-empty column names.
        strict: Whether adjacent values must be strictly increasing.

    Returns:
        A table check that reports failing rows.

    Raises:
        ValueError: If fewer than two columns are given, a name is empty, or
            names repeat.
    """

def sum_columns(addends: list[str], target: str, tolerance: float | None = None) -> DataFrameCheck:
    """Require ``target == sum(addends)`` in every row.

    Integer columns with no float or mixed signedness are summed exactly.
    Any float column (or a mix of signed/unsigned integers) switches the row
    to ``f64`` arithmetic, where ``tolerance`` applies. Null operands skip
    the row.

    Args:
        addends: One or more column names to sum.
        target: Column that must equal the sum.
        tolerance: Non-negative absolute tolerance for floating-point sums.
            Defaults to exact comparison.

    Returns:
        A table check that reports failing rows.

    Raises:
        ValueError: If ``addends`` is empty or ``tolerance`` is negative or
            non-finite.
    """

def implies(left: str, by: str) -> DataFrameCheck:
    """Require presence implication: non-null ``left`` needs non-null ``by``.

    Rows with a null ``left`` never fail. Rows with a non-null ``left`` and a
    null ``by`` are reported.

    Args:
        left: Column whose presence triggers the requirement.
        by: Column that must also be present.

    Returns:
        A table check that reports failing rows.

    Raises:
        ValueError: If ``left == by``.
    """

def determines(a: str, b: str) -> DataFrameCheck:
    """Require ``a`` to functionally determine ``b``.

    The first non-null ``b`` seen for each ``a`` is canonical; every row whose
    ``b`` disagrees is reported together with the canonical row. Null ``a`` or
    null ``b`` cells do not participate.

    Args:
        a: Determinant column.
        b: Dependent column.

    Returns:
        A table check that reports violating rows.

    Raises:
        ValueError: If ``a == b``.
    """

def within(left: str, right: str, max_gap: timedelta) -> DataFrameCheck:
    """Require a temporal gap between ``left`` and ``right``.

    Both columns must be temporal and share the exact same dtype. The gap is
    measured in the dtype's natural unit (days for ``Date``, nanoseconds for
    ``Datetime``/``Time``). The bound is inclusive, and nulls skip the row.

    Args:
        left: Earlier column name.
        right: Later column name.
        max_gap: Maximum allowed gap.

    Returns:
        A table check that reports failing rows.

    Raises:
        ValueError: If ``left == right`` or ``max_gap`` exceeds the storable
            nanosecond range.
    """

def infer_schema(df: Any) -> Schema:
    """Draft a schema from a Polars ``DataFrame`` or ``LazyFrame``.

    Inference is a first draft: it captures dtypes, nullability, observed
    bounds, known string patterns, small membership sets, uniqueness, and list
    lengths. Review and tighten the result before persisting it.

    Args:
        df: Frame to inspect.

    Returns:
        A schema with inferred columns and no table checks.
    """

def sample(schema: Schema, n: int = 10, seed: int | None = None) -> Any:
    """Generate a valid frame from ``schema``.

    Equivalent to ``schema.sample(n, seed)``. Generation honors the schema's
    checks, including cross-column rules, and raises rather than returning
    data that would fail validation.

    Args:
        schema: Schema to generate from.
        n: Number of rows.
        seed: Optional RNG seed for reproducible output.

    Returns:
        A ``polars.DataFrame`` satisfying the schema.

    Raises:
        ValueError: If the schema is contradictory or cannot be synthesized.
    """

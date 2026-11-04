"""Decorator utilities to attach schema validation to Python functions.

:func:`guard` wraps a callable and validates the DataFrame arguments you
name (and optionally the returned DataFrame) against a
:class:`~framelaw._framelaw.Schema` before control enters or leaves the
function.
"""

from __future__ import annotations

import functools
import inspect
from collections.abc import Callable, Sequence
from typing import Any

from ._framelaw import Schema, SchemaError

__all__ = ["guard"]


def _validate_target(
    schema: Schema,
    df: Any,
    what: str,
    chunk_rows: int | None,
) -> None:
    """Validates one DataFrame, raising ``SchemaError`` with context."""
    report = schema.validate(df, raise_on_error=False, chunk_rows=chunk_rows)
    if report is not None:
        raise SchemaError(f"{what} failed schema validation:\n{report.summary()}")


def guard(
    schema: Schema,
    df: str | Sequence[str] = "df",
    *,
    returns: Schema | None = None,
    chunk_rows: int | None = None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Validate DataFrame arguments (and optionally the return value).

    The decorated function validates every argument whose parameter name
    is listed in ``df`` against ``schema`` before it runs, and — when
    ``returns`` is given — the returned DataFrame against that schema
    before the caller sees it. Validation failures raise
    :class:`~framelaw.SchemaError`; the message names the failing
    argument.

    Args:
        schema: Schema applied to the named argument(s).
        df: Parameter name or names of the DataFrame argument(s) to
            validate. Keyword arguments are supported. ``self`` and other
            non-DataFrame parameters are never touched.
        returns: Optional schema applied to the decorated function's
            return value.
        chunk_rows: Optional batch size forwarded to
            :meth:`Schema.validate`. Dataset-wide checks
            (``unique``, ``unique_together``, ``determines``) only see one
            batch at a time.

    Returns:
        A decorator wrapping the function with the validations described
        above. Async functions (including methods) are supported; static
        and class methods must be decorated below their
        ``@staticmethod`` / ``@classmethod`` wrapper.

    Raises:
        ValueError: At decoration time if a ``df`` name is not a
            parameter of the decorated callable, or is a duplicated name.

    Examples:
        >>> import polars as pl
        >>> import framelaw as fl
        >>> users = fl.Schema({"user_id": fl.Column(pl.Int64, nullable=False)})
        >>> @fl.guard(users, df="df")
        ... def process(df: pl.DataFrame) -> pl.DataFrame:
        ...     return df
        >>> _ = process(pl.DataFrame({"user_id": [1, 2]}))
    """
    parameter_names = (df,) if isinstance(df, str) else tuple(df)
    if not parameter_names:
        raise ValueError("guard requires at least one DataFrame parameter name")
    if len(set(parameter_names)) != len(parameter_names):
        raise ValueError(f"guard received duplicate parameter names: {parameter_names}")

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        """Wraps ``func`` with the argument/return validations."""
        callable_name = getattr(func, "__name__", repr(func))
        signature = inspect.signature(func)
        for name in parameter_names:
            if name not in signature.parameters:
                raise ValueError(
                    f"guard parameter '{name}' is not an argument of "
                    f"{callable_name!r} (parameters: {list(signature.parameters)})"
                )

        # Precompute what the usual call shape needs so the common case
        # validates from `args` directly: ``Signature.bind`` (plus
        # ``apply_defaults``) costs far more than the validation itself
        # for small frames.
        parameters = list(signature.parameters.values())
        has_var_kind = any(
            parameter.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
            for parameter in parameters
        )
        positional_names = [
            parameter.name
            for parameter in parameters
            if parameter.kind
            in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        ]
        df_positions = tuple(
            (name, positional_names.index(name))
            for name in parameter_names
            if name in positional_names
        )
        fast_call_allowed = not has_var_kind and len(df_positions) == len(parameter_names)

        def validate_arguments(args: tuple[Any, ...]) -> None:
            """Validates the prepared positional arguments."""
            for name, index in df_positions:
                _validate_target(schema, args[index], f"Argument '{name}'", chunk_rows)

        def validate_bound(args: tuple[Any, ...], passed_kwargs: dict[str, Any]) -> None:
            """Binds arguments with defaults, then validates df parameters."""
            bound = signature.bind(*args, **passed_kwargs)
            bound.apply_defaults()
            for name in parameter_names:
                _validate_target(schema, bound.arguments[name], f"Argument '{name}'", chunk_rows)

        if inspect.iscoroutinefunction(func):

            @functools.wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                if fast_call_allowed and not kwargs and len(args) == len(positional_names):
                    validate_arguments(args)
                    result = await func(*args)
                else:
                    validate_bound(args, kwargs)
                    result = await func(*args, **kwargs)
                if returns is not None:
                    _validate_target(returns, result, "Return value", chunk_rows)
                return result

            return async_wrapper

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if fast_call_allowed and not kwargs and len(args) == len(positional_names):
                validate_arguments(args)
                result = func(*args)
            else:
                validate_bound(args, kwargs)
                result = func(*args, **kwargs)
            if returns is not None:
                _validate_target(returns, result, "Return value", chunk_rows)
            return result

        return wrapper

    return decorator

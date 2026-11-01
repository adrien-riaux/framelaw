import asyncio

import framelaw as fl
import polars as pl
import pytest


@pytest.fixture()
def users_schema() -> fl.Schema:
    """Require a positive ``user_id`` column."""
    return fl.Schema(
        {
            "user_id": fl.Column(pl.Int64, nullable=False, checks=[fl.checks.column.gt(0)]),
        }
    )


@pytest.fixture()
def valid_frame() -> pl.DataFrame:
    """A frame satisfying ``users_schema``."""
    return pl.DataFrame({"user_id": [1, 2, 3]})


@pytest.fixture()
def invalid_frame() -> pl.DataFrame:
    """A frame violating ``users_schema``."""
    return pl.DataFrame({"user_id": [1, -2, 3]})


def test_guard_validates_named_argument(users_schema: fl.Schema, valid_frame: pl.DataFrame) -> None:
    """Pass valid frames through untouched."""

    @fl.guard(users_schema, df="df")
    def process(df: pl.DataFrame) -> pl.DataFrame:
        return df.with_columns(pl.col("user_id") * 10)

    assert process(valid_frame)["user_id"].to_list() == [10, 20, 30]


def test_guard_raises_schema_error_with_argument_context(
    users_schema: fl.Schema, invalid_frame: pl.DataFrame
) -> None:
    """Failures raise ``SchemaError`` naming the failing argument."""

    @fl.guard(users_schema, df="df")
    def process(df: pl.DataFrame) -> pl.DataFrame:
        return df

    with pytest.raises(fl.SchemaError, match="Argument 'df' failed schema validation"):
        process(invalid_frame)
    with pytest.raises(ValueError):  # SchemaError stays catchable broadly
        process(invalid_frame)


def test_guard_validates_keyword_arguments(
    users_schema: fl.Schema, invalid_frame: pl.DataFrame
) -> None:
    """Keyword-bound arguments are validated too."""

    @fl.guard(users_schema, df="frame")
    def process(*, frame: pl.DataFrame) -> pl.DataFrame:
        return frame

    with pytest.raises(fl.SchemaError, match="Argument 'frame'"):
        process(frame=invalid_frame)


def test_guard_validates_multiple_named_parameters(
    users_schema: fl.Schema,
    valid_frame: pl.DataFrame,
    invalid_frame: pl.DataFrame,
) -> None:
    """Every named parameter is checked, in declaration order."""

    @fl.guard(users_schema, df=["first", "second"])
    def merge(first: pl.DataFrame, second: pl.DataFrame) -> pl.DataFrame:
        return first

    assert merge(valid_frame, valid_frame).equals(valid_frame)
    with pytest.raises(fl.SchemaError, match="Argument 'second'"):
        merge(valid_frame, invalid_frame)


def test_guard_validates_return_value(
    users_schema: fl.Schema,
    invalid_frame: pl.DataFrame,
    valid_frame: pl.DataFrame,
) -> None:
    """``returns=schema`` checks the outgoing DataFrame."""

    @fl.guard(users_schema, df="df", returns=users_schema)
    def transform(df: pl.DataFrame) -> pl.DataFrame:
        return df.with_columns(pl.col("user_id") - 2)

    with pytest.raises(fl.SchemaError, match="Return value failed schema validation"):
        transform(valid_frame)


def test_guard_supports_methods(
    users_schema: fl.Schema,
    valid_frame: pl.DataFrame,
    invalid_frame: pl.DataFrame,
) -> None:
    """Instance methods validate their DataFrame argument, ``self`` untouched."""

    class Service:
        @fl.guard(users_schema, df="df")
        def process(self, df: pl.DataFrame) -> pl.DataFrame:
            return df

    service = Service()
    assert service.process(valid_frame).equals(valid_frame)
    with pytest.raises(fl.SchemaError):
        service.process(invalid_frame)


def test_guard_supports_staticmethod_and_classmethod(
    users_schema: fl.Schema,
    valid_frame: pl.DataFrame,
    invalid_frame: pl.DataFrame,
) -> None:
    """Decorators compose below ``@staticmethod`` / ``@classmethod``."""

    class Service:
        @staticmethod
        @fl.guard(users_schema, df="df")
        def static_process(df: pl.DataFrame) -> pl.DataFrame:
            return df

        @classmethod
        @fl.guard(users_schema, df="df")
        def class_process(cls, df: pl.DataFrame) -> pl.DataFrame:
            return df

    assert Service.static_process(valid_frame) is not None
    with pytest.raises(fl.SchemaError):
        Service.static_process(invalid_frame)
    assert Service.class_process(valid_frame) is not None
    with pytest.raises(fl.SchemaError):
        Service.class_process(invalid_frame)


def test_guard_supports_async_functions(
    users_schema: fl.Schema,
    valid_frame: pl.DataFrame,
    invalid_frame: pl.DataFrame,
) -> None:
    """Coroutine functions validate arguments and behave like the sync path."""

    async def run() -> None:
        @fl.guard(users_schema, df="df")
        async def process(df: pl.DataFrame) -> pl.DataFrame:
            return df

        assert await process(valid_frame) is not None
        with pytest.raises(fl.SchemaError):
            await process(invalid_frame)

    asyncio.run(run())


def test_guard_unknown_parameter_fails_at_decoration(
    users_schema: fl.Schema, valid_frame: pl.DataFrame
) -> None:
    """Naming a non-existent parameter raises ``ValueError`` immediately."""
    with pytest.raises(ValueError, match="'frame' is not an argument"):

        @fl.guard(users_schema, df="frame")
        def process(df: pl.DataFrame) -> pl.DataFrame:
            return df


def test_guard_rejects_duplicate_parameters(users_schema: fl.Schema) -> None:
    """Duplicated parameter names are rejected at construction."""
    with pytest.raises(ValueError, match="duplicate parameter names"):

        @fl.guard(users_schema, df=["df", "df"])
        def process(df: pl.DataFrame) -> pl.DataFrame:
            return df


def test_guard_rejects_empty_parameter_list(users_schema: fl.Schema) -> None:
    """An empty parameter list cannot validate anything."""
    with pytest.raises(ValueError, match="at least one"):

        @fl.guard(users_schema, df=[])
        def process(df: pl.DataFrame) -> pl.DataFrame:
            return df


def test_guard_non_dataframe_argument_raises_type_error(users_schema: fl.Schema) -> None:
    """Passing a non-frame to a guarded argument surfaces ``TypeError``."""

    @fl.guard(users_schema, df="df")
    def process(df: pl.DataFrame) -> pl.DataFrame:
        return df

    with pytest.raises(TypeError):
        process("not a frame")  # pyright: ignore[reportArgumentType]


def test_guard_preserves_function_metadata(
    users_schema: fl.Schema, valid_frame: pl.DataFrame
) -> None:
    """``functools.wraps`` keeps name and docstring."""

    @fl.guard(users_schema, df="df")
    def process(df: pl.DataFrame) -> pl.DataFrame:
        """Original docstring."""
        return df

    assert process.__name__ == "process"
    assert process.__doc__ == "Original docstring."


def test_guard_passes_chunk_rows(users_schema: fl.Schema, valid_frame: pl.DataFrame) -> None:
    """``chunk_rows`` forwards to batched validation."""

    @fl.guard(users_schema, df="df", chunk_rows=2)
    def process(df: pl.DataFrame) -> pl.DataFrame:
        return df

    assert process(valid_frame) is not None

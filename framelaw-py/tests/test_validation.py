import framelaw as fl
import polars as pl
import pytest


def test_validation_returns_report_with_rows() -> None:
    """Return structured failures when raising is disabled."""
    schema = fl.Schema(
        {
            "age": fl.Column(pl.Int64, nullable=False, checks=[fl.gt(0)]),
            "name": fl.Column(pl.String, nullable=False),
        }
    )
    dataframe = pl.DataFrame({"age": [1, -2, 3], "name": ["a", None, "c"]})

    report = schema.validate(dataframe, raise_on_error=False)

    assert not report.is_valid()
    assert any(error.column == "age" and error.rows == [1] for error in report.errors)
    assert any(error.column == "name" and error.rows == [1] for error in report.errors)
    with pytest.raises(ValueError, match="Validation failed"):
        schema.validate(dataframe)


def test_aliases_and_valid_data() -> None:
    """Expose the documented check aliases and accept valid data."""
    schema = fl.Schema(
        {
            "country": fl.Column(
                pl.String,
                checks=[fl.one_of(["FR", "US"]), fl.matches(r"^[A-Z]{2}$")],
            ),
            "id": fl.Column(pl.Int64, checks=[fl.unique()]),
        }
    )
    dataframe = pl.DataFrame({"country": ["FR", "US"], "id": [1, 2]})

    assert schema.validate(dataframe) is None


def test_invalid_regex_is_a_python_error() -> None:
    """Reject malformed regexes during check construction."""
    with pytest.raises(ValueError, match="Invalid regular expression"):
        fl.matches("[")


def test_dtype_mismatch_is_reported() -> None:
    """Reject a DataFrame column whose native dtype differs from the schema."""
    schema = fl.Schema({"value": fl.Column(pl.Int32)})
    dataframe = pl.DataFrame({"value": [1, 2]})

    report = schema.validate(dataframe, raise_on_error=False)

    assert not report.is_valid()
    assert report.errors[0].check == "Dtype"
    assert report.errors[0].rows == []

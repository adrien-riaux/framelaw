import polars as pl
from framelaw import Column, Schema, gt, not_null


def main():
    schema = Schema(
        {
            "age": Column(dtype=pl.Float64, nullable=False, checks=[gt(0.0)]),
            "name": Column(dtype=pl.String, nullable=False, checks=[not_null()]),
        }
    )

    valid_df = pl.DataFrame(
        {"age": [10.0, 20.0, 30.0], "name": ["Alice", "Bob", "Dave"]},
        schema={"age": pl.Float64, "name": pl.String},
    )
    assert schema.validate(valid_df) is None

    invalid_df = pl.DataFrame(
        {"age": [10.0, 20.0, -5.0, 30.0], "name": ["Alice", "Bob", None, "Dave"]}
    )
    report = schema.validate(invalid_df, raise_on_error=False)
    assert not report.is_valid()
    assert any(error.column == "age" and error.rows == [2] for error in report.errors)
    assert any(error.column == "name" and error.rows == [2] for error in report.errors)
    print(report.summary())

    try:
        schema.validate(invalid_df)
    except ValueError as error:
        print("\nValidation raised as expected:")
        print(error)
    else:
        raise AssertionError("invalid data should raise by default")


if __name__ == "__main__":
    main()

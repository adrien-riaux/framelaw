import polars as pl
from framelaw import Schema, Column, gt, not_null


def main():
    # 1. Create a Schema using the exposed classes
    # In Rust, these wrap the core validation logic.
    schema = Schema(
        {
            "age": Column(dtype="float64", nullable=False, checks=[gt(0.0)]),
            "name": Column(dtype="str", nullable=False, checks=[not_null()]),
        }
    )

    print("Schema created successfully!")

    # 2. Create a Polars DataFrame with some bad data
    df = pl.DataFrame(
        {"age": [10.0, 20.0, -5.0, 30.0], "name": ["Alice", "Bob", None, "Dave"]}
    )
    print("\nDataFrame:")
    print(df)

    # 3. Validate
    # This invokes the PyO3 binding, translating the DataFrame zero-copy
    # to Rust, and running the core validation engine.
    try:
        schema.validate(df)
        print("\nValidation Result: Passed (Unexpected for this data)")
    except ValueError as e:
        print("\nValidation failed as expected. Error Report From Rust:")
        print(e)
    except Exception as e:
        print(
            "\nValidation raised an unexpected exception (likely due to pending Windows MSVC build environment fixes):"
        )
        print(e)


if __name__ == "__main__":
    main()

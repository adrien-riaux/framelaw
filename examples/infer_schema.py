"""Infer a schema from trusted data, refine it, then enforce it."""

import polars as pl
from framelaw import infer_schema, lt


def main() -> None:
    """Run the infer, refine, and enforce workflow."""
    trusted = pl.DataFrame(
        {
            "age": [25, 31, 42],
            "email": ["a@example.com", "b@example.com", "c@example.com"],
        },
        schema={"age": pl.Int64, "email": pl.String},
    )

    schema = infer_schema(trusted)
    print(schema.to_json())

    schema["age"].add_check(lt(120))

    assert schema.validate(trusted) is None
    print("trusted data passes")

    new_data = pl.DataFrame(
        {
            "age": [25, 200, 42],
            "email": ["a@example.com", "nope", "c@example.com"],
        },
        schema={"age": pl.Int64, "email": pl.String},
    )
    report = schema.validate(new_data, raise_on_error=False)
    assert report is not None and not report.is_valid()
    print(report.summary())


if __name__ == "__main__":
    main()

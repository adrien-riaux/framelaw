"""Derive a FrameLaw schema from a Pydantic model."""

import polars as pl
from framelaw import Schema
from pydantic import BaseModel, Field


class User(BaseModel):
    """User record with validated fields."""

    user_id: int
    age: int = Field(gt=0, lt=120)
    email: str = Field(pattern=r".+@.+\..+")
    country: str | None = None


def main() -> None:
    """Validate a DataFrame against the derived schema."""
    schema = Schema.from_pydantic(User)

    good = pl.DataFrame(
        {
            "user_id": [1, 2],
            "age": [28, 35],
            "email": ["a@example.com", "b@example.com"],
            "country": ["FR", None],
        }
    )
    assert schema.validate(good) is None
    print("good data passes")

    bad = pl.DataFrame(
        {
            "user_id": [1, 2],
            "age": [28, -4],
            "email": ["a@example.com", "nope"],
            "country": ["FR", None],
        }
    )
    report = schema.validate(bad, raise_on_error=False)
    assert report is not None and not report.is_valid()
    print(report.summary())


if __name__ == "__main__":
    main()

# Pydantic integration

If you already model your data with Pydantic v2, derive a FrameLaw schema
directly from the `BaseModel` — field types and `Field(...)` constraints
translate to columns and checks automatically.

```python
from typing import Optional
from pydantic import BaseModel, Field
import framelaw as fl

class User(BaseModel):
    user_id: int
    age: int = Field(gt=0, lt=120)
    email: str = Field(pattern=r".+@.+\..+")
    score: float = Field(ge=0.0, le=1.0)
    country: Optional[str] = None

schema = fl.Schema.from_pydantic(User)
```

FrameLaw reads the model's `__pydantic_core_schema__` natively, without
Python reflection overhead.

## Type mapping

| Pydantic | FrameLaw dtype | Checks |
|---|---|---|
| `int` | `pl.Int64` | `not_null()` |
| `float` | `pl.Float64` | `not_null()` |
| `str` | `pl.String` | `not_null()` |
| `bool` | `pl.Boolean` | `not_null()` |
| `Optional[T]` | dtype of `T` | `nullable=True` |
| `Field(gt=n)` | — | `gt(n)` |
| `Field(lt=n)` | — | `lt(n)` |
| `Field(ge=n)` | — | `gte(n)` |
| `Field(le=n)` | — | `lte(n)` |
| `Field(pattern=...)` | — | `matches(...)` |
| `Field(min_length=m, max_length=n)` on `list` | — | `list_len_between(m, n)` |
| `Literal["a", "b"]` | `pl.String` | `one_of(["a", "b"])` |
| `list[T]` (`T` scalar) | `pl.List` of the mapped inner dtype | length checks as above |

Unsupported field types (e.g. `dict`, nested models) raise `TypeError`
rather than silently producing a weaker schema.

> `min_length`/`max_length` on `str` fields are currently ignored (there is
> no string-length check yet); they only take effect on `list` fields.

> FrameLaw targets Pydantic v2 only; models must expose
> `__pydantic_core_schema__`.

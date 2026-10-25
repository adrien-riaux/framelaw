"""Execute runnable Python snippets from the documentation.

The docs contain a few illustrative snippets that depend on external files or
intentionally elide code with ``...``. Those are skipped; everything else must
run against the installed extension.
"""

from __future__ import annotations

import os
import re
import sys
import tempfile
import types
from pathlib import Path
from typing import Any

SNIPPET = re.compile(r"```python\n(.*?)```", re.DOTALL)
SKIP_MARKERS = ("...", "read_csv", "read_parquet", "scan_csv", "scan_parquet")


def _blocks(path: Path) -> list[str]:
    """Return the Python code fences in one Markdown file."""
    return SNIPPET.findall(path.read_text(encoding="utf-8"))


def main() -> None:
    """Run every non-skipped Python snippet and fail on the first error."""
    docs = Path(__file__).resolve().parent.parent / "docs"
    checked = 0
    skipped = 0
    failures: list[str] = []
    previous = Path.cwd()
    with tempfile.TemporaryDirectory() as tmp:
        os.chdir(tmp)
        try:
            for path in sorted(docs.rglob("*.md")):
                module_name = f"_framelaw_docs_{path.stem.replace('-', '_')}"
                module = types.ModuleType(module_name)
                module.__file__ = str(path)
                namespace: dict[str, Any] = module.__dict__
                sys.modules[module_name] = module
                exec("import polars as pl\nimport framelaw as fl\n", namespace)  # noqa: S102
                for index, block in enumerate(_blocks(path), 1):
                    if any(marker in block for marker in SKIP_MARKERS):
                        skipped += 1
                        continue
                    try:
                        exec(compile(block, f"{path.name}#{index}", "exec"), namespace)  # noqa: S102
                    except Exception as error:  # noqa: BLE001
                        relative = path.relative_to(docs)
                        failures.append(
                            f"{relative}#{index}: {type(error).__name__}: {error}"
                        )
                    else:
                        checked += 1
                sys.modules.pop(module_name, None)
        finally:
            os.chdir(previous)
    if failures:
        raise SystemExit("Docs snippets failed:\n" + "\n".join(failures))
    print(f"docs snippets: {checked} executed, {skipped} skipped")


if __name__ == "__main__":
    main()

# Contributing to FrameLaw

Thanks for contributing. This guide covers setup, workflow, and the gates
every change must pass.

## Setup

You need Python 3.10+, `uv`, and a Rust toolchain (stable).

```bash
git clone <repo-url> && cd framelaw
uv venv
make build   # editable install of the extension (debug profile)
make test    # confirm green before changing anything
```

## Workflow

1. Open an issue or pick one; keep changes focused and step-by-step.
2. Read `AGENTS.md` (agent instructions, module map, command reference) and
   `docs/architecture.md` (design rules) before touching code.
3. Implement with tests: Rust unit tests next to the module, `pytest`
   coverage in `framelaw-py/tests/` for any public API change.
4. Update the user-facing surface alongside the code: `docs/` pages,
   `docs/api-reference.md`, `framelaw-py/python/framelaw/_framelaw.pyi`
   (native signatures), and `CHANGELOG.md` (newest entry on top).
5. Run `make lint`, `make test`, and `make scripts` locally. Open a PR against `dev`; CI runs the same gate
   plus the Python 3.10–3.14 matrix.

## Standards

- **Rust**: `cargo fmt`, `cargo clippy --all-targets -- -D warnings`.
  Idiomatic, allocation-aware code; borrows over clones.
- **Python**: `ruff check`, `ruff format --check`, `ty check` clean,
  `interrogate` at 100% for `framelaw-py/python` and `scripts/`.
- **Separation**: `framelaw-core` must never import PyO3/Python types.
- **New checks** require: core struct + `CheckSpec` variant, Python
  constructor, `JsonCheck` round-trip, stub entry, `__init__.py` export,
  docs, and tests on both sides.
- **Schemas**: JSON stays backward-compatible (current writer is v2, loader
  accepts v1+). Scripts in `scripts/` are capability-named, runnable,
  self-asserting, and must exit 0. Every docs snippet must execute against
  the built extension before committing.

## Performance

Editable installs compile Rust in **debug** mode. Never quote benchmark
numbers from one — run `make build-release` first and use
`scripts/benchmark.py`. Published wheels always build with `--release`.

## Releases

Maintainers release from version tags; CI handles the rest:

- `vX.Y.Z-rc.N` → full gate, then TestPyPI (canary).
- `vX.Y.Z` → full gate, then PyPI plus a GitHub release with artifacts.

Both use PyPI trusted publishing (no tokens): the repo must be registered
as a trusted publisher on PyPI and TestPyPI first. Keep the three version
declarations in sync: `framelaw-py/pyproject.toml` and both `Cargo.toml`
files.

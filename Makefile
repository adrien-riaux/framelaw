.PHONY: help format lint lint-py typecheck docstyle lint-all check build build-release wheel publish test test-rust test-py example examples benchmark ci all clean

help:
	@echo "Available commands:"
	@echo "  make format     - Auto-format Rust and Python code"
	@echo "  make lint       - Clippy (Rust) + ruff check (Python)"
	@echo "  make lint-py    - ruff check + ruff format check (verify only)"
	@echo "  make typecheck  - Type-check Python with ty"
	@echo "  make docstyle   - Docstring coverage with interrogate"
	@echo "  make lint-all   - All linters: clippy, ruff, ty, interrogate"
	@echo "  make check      - Verify-only gate (same as lint-all)"
	@echo "  make build      - Build the Rust extension, editable debug install"
	@echo "  make build-release - Rebuild the extension optimized (--release)"
	@echo "  make wheel      - Release wheel + sdist into ./dist"
	@echo "  make publish    - Upload ./dist to TestPyPI (PYPI_URL=... to override)"
	@echo "  make test       - Rust + Python test suites (test-rust, test-py)"
	@echo "  make example    - Run the basic example"
	@echo "  make benchmark  - Run the throughput benchmark"
	@echo "  make examples   - Run every example script"
	@echo "  make ci         - Full gate: lint-all, build, test, examples"
	@echo "  make all        - check, build, test"
	@echo "  make clean      - Clean the build artifacts"

format:
	@echo "=> Formatting Rust code..."
	cargo fmt
	@echo "=> Formatting Python code..."
	uvx ruff format .

lint:
	@echo "=> Linting Rust code with Clippy..."
	cargo clippy --all-targets -- -D warnings
	@echo "=> Linting Python code..."
	uvx ruff check framelaw-py examples

lint-py:
	@echo "=> Checking Python lint + format..."
	uvx ruff check framelaw-py examples
	uvx ruff format --check framelaw-py examples

typecheck:
	@echo "=> Type-checking Python code with ty..."
	uv run --with ty ty check framelaw-py/python examples

docstyle:
	@echo "=> Checking docstring coverage with interrogate..."
	uvx interrogate framelaw-py/python examples

lint-all: lint typecheck docstyle

check: lint-all

build:
	@echo "=> Building Python extension (debug)..."
	uv pip install -e ./framelaw-py

build-release:
	@echo "=> Building optimized release extension..."
	VIRTUAL_ENV=$${VIRTUAL_ENV:-$$(pwd)/.venv} uvx maturin develop --release --manifest-path framelaw-py/Cargo.toml

wheel:
	@echo "=> Building release wheel + sdist into ./dist..."
	uvx maturin build --release --manifest-path framelaw-py/Cargo.toml --out dist

publish: wheel
	@echo "=> Uploading ./dist (default: TestPyPI)..."
	uvx twine upload --repository-url $${PYPI_URL:-https://test.pypi.org/legacy/} dist/*

test: test-rust test-py

test-rust:
	@echo "=> Running Rust tests..."
	cargo test -p framelaw-core

test-py:
	@echo "=> Running Python tests..."
	uv run --with-editable ./framelaw-py pytest framelaw-py/tests -q

example:
	@echo "=> Running basic example..."
	uv run --with-editable ./framelaw-py python examples/basic_validation.py

benchmark:
	@echo "=> Running benchmark (numbers only meaningful on build-release)..."
	uv run --with-editable ./framelaw-py python examples/benchmark.py

examples:
	@echo "=> Running all examples..."
	uv run --with-editable ./framelaw-py python examples/basic_validation.py
	uv run --with-editable ./framelaw-py python examples/infer_schema.py
	uv run --with-editable ./framelaw-py python examples/pydantic_model.py
	uv run --with-editable ./framelaw-py python examples/error_report.py
	uv run --with-editable ./framelaw-py python examples/cross_column_nested.py
	uv run --with-editable ./framelaw-py python examples/lazy_sampling.py
	uv run --with-editable ./framelaw-py python examples/generate_data.py

ci: lint-all build test examples

all: check build test

clean:
	cargo clean
	rm -rf .venv dist

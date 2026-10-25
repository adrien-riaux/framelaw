.PHONY: help format lint build build-release test scripts docs-serve docs-build docs-check wheel publish clean

help:
	@echo "Available commands:"
	@echo "  make format        - Auto-format Rust and Python code"
	@echo "  make lint          - Full verify: clippy, ruff, ty, interrogate"
	@echo "  make build         - Build the extension, editable debug install"
	@echo "  make build-release - Rebuild the extension optimized (--release)"
	@echo "  make test          - Rust + Python test suites"
	@echo "  make scripts       - Run every smoke-test / benchmark script"
	@echo "  make docs-serve    - Live-preview the documentation site"
	@echo "  make docs-build    - Strict documentation build into ./site"
	@echo "  make docs-check    - Strict build + execute runnable docs snippets"
	@echo "  make wheel         - Release wheel + sdist into ./dist"
	@echo "  make publish       - Upload ./dist to TestPyPI (PYPI_URL=... to override)"
	@echo "  make clean         - Clean the build artifacts"

format:
	@echo "=> Formatting Rust code..."
	cargo fmt
	@echo "=> Formatting Python code..."
	uvx ruff format framelaw-py scripts

lint:
	@echo "=> Linting Rust code with Clippy..."
	cargo clippy --all-targets -- -D warnings
	@echo "=> Linting Python code..."
	uvx ruff check framelaw-py scripts
	uvx ruff format --check framelaw-py scripts
	@echo "=> Type-checking Python code with ty..."
	test -d .venv || uv venv
	uv pip install -q ty polars pydantic
	uv run ty check --extra-search-path framelaw-py/python framelaw-py/python scripts
	@echo "=> Checking docstring coverage with interrogate..."
	uvx interrogate framelaw-py/python scripts

build:
	@echo "=> Building Python extension (debug, with dev extras)..."
	uv pip install -e "./framelaw-py[dev]"

build-release:
	@echo "=> Building optimized release extension..."
	VIRTUAL_ENV=$${VIRTUAL_ENV:-$$(pwd)/.venv} uvx maturin develop --release --manifest-path framelaw-py/Cargo.toml

wheel:
	@echo "=> Building release wheel + sdist into ./dist..."
	uvx maturin build --release --manifest-path framelaw-py/Cargo.toml --out dist
	uvx maturin sdist --manifest-path framelaw-py/Cargo.toml --out dist

publish: wheel
	@echo "=> Uploading ./dist (default: TestPyPI)..."
	uvx twine upload --repository-url $${PYPI_URL:-https://test.pypi.org/legacy/} dist/*

test:
	@echo "=> Running Rust tests..."
	cargo test -p framelaw-core
	@echo "=> Running Python tests..."
	uv run --with pytest --with-editable ./framelaw-py pytest framelaw-py/tests -q

scripts:
	@echo "=> Running all scripts..."
	for script in scripts/*.py; do \
		uv run --with-editable ./framelaw-py python "$$script"; \
	done

docs-serve:
	@echo "=> Serving documentation at http://127.0.0.1:8000..."
	uvx --with-requirements docs/assets/requirements.txt mkdocs serve

docs-build:
	@echo "=> Building documentation (strict)..."
	uvx --with-requirements docs/assets/requirements.txt mkdocs build --strict

docs-check: docs-build
	@echo "=> Executing runnable documentation snippets..."
	uv run --with-editable ./framelaw-py python scripts/check_docs.py

clean:
	cargo clean
	rm -rf .venv dist

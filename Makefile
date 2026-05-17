.PHONY: help format lint check build test example all clean

help:
	@echo "Available commands:"
	@echo "  make format  - Auto-format Rust and Python code"
	@echo "  make lint    - Run Linters for Rust (Clippy) and Python (Ruff)"
	@echo "  make check   - Run formatting and linting checks"
	@echo "  make build   - Build the Rust extension for Python using Maturin/uv"

format:
	@echo "=> Formatting Rust code..."
	cargo fmt
	@echo "=> Formatting Python code..."
	uvx ruff format .

lint:
	@echo "=> Linting Rust code with Clippy..."
	cargo clippy --all-targets -- -D warnings
	@echo "=> Linting Python code..."
	uvx ruff check .

check: format lint

build:
	@echo "=> Building Python extension..."
	uv pip install -e ./framelaw-py


# Installation

FrameLaw is a native Rust extension with prebuilt wheels for the common
platforms. Installing it does **not** require a Rust toolchain.

## Requirements

- Python **3.10+** (CPython; the wheels use the limited API, `abi3-py310`)
- [`polars`](https://pola.rs/) **1.0+**
- [`pydantic`](https://docs.pydantic.dev/) **2.x** (installed as a dependency;
  used by `Schema.from_pydantic`)

Both are declared as runtime dependencies, so a normal install pulls them in.
The native extension is loaded directly by the `framelaw` package.

## Install from PyPI

With [uv](https://docs.astral.sh/uv/):

```bash
uv add framelaw
```

With pip:

```bash
pip install framelaw
```

Verify the install:

```bash
python -c "import framelaw; print(framelaw.__version__)"
```

## Supported platforms

Prebuilt wheels are published for:

| Platform | Architectures |
|---|---|
| macOS | Apple Silicon (`arm64`), Intel (`x86_64`) |
| Linux | `x86_64`, `aarch64` |
| Windows | `x86_64` |

Because the extension targets `abi3-py310`, one wheel covers CPython 3.10
through 3.14 on each platform. If your platform has no wheel, pip builds from
source and needs a stable Rust toolchain.

## Build from source

```bash
git clone https://github.com/adrien-riaux/framelaw.git
cd framelaw
uv venv
make build          # editable debug install
make build-release  # optimized build for benchmarks
```

`make build` installs the package in editable mode with the development
extras. Editable installs compile Rust in **debug** mode: fast to rebuild,
slower at runtime. Published wheels and `make build-release` use `--release`.

## Documentation toolchain

The docs site is built with MkDocs Material and mkdocstrings. It needs Python
3.11+ for the docs dependencies, independently of the package's 3.10 floor:

```bash
make docs-serve   # live preview at http://127.0.0.1:8000
make docs-build   # strict build into ./site
make docs-check   # strict build + execute runnable snippets
```

The docs build parses the Python package and its `.pyi` stub statically, so it
does **not** compile the Rust extension.

## Next steps

- [Quickstart](quickstart.md) — validate a frame in five minutes.
- [Why FrameLaw](why-framelaw.md) — design principles and trade-offs.
- [API reference](reference/index.md) — every class and constructor.

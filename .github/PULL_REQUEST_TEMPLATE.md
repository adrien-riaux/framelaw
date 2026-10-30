## Summary

<!-- What does this change, and why? -->

Closes #

## Type of change

- [ ] Bug fix
- [ ] New feature or check
- [ ] Documentation
- [ ] Refactor or performance
- [ ] CI or tooling

## Checklist

- [ ] `make lint` passes
- [ ] `make test` passes
- [ ] `make scripts` passes (when scripts are affected)
- [ ] `make docs-check` passes (when docs or the public API change)
- [ ] User-facing changes are documented in `docs/` and `CHANGELOG.md`
- [ ] `_framelaw.pyi` matches `framelaw-py/src/lib.rs`, with Google-style
      docstrings for every new symbol
- [ ] Schema JSON remains backward compatible (writes v2, still loads v1)
- [ ] Versions stay in sync across `framelaw-py/pyproject.toml` and both
      `Cargo.toml` files (release PRs only)

### New checks

<!-- Remove this section if the PR does not add a check. -->

- [ ] Core struct and `CheckSpec` variant
- [ ] `check_from_spec` support (when the check is nestable)
- [ ] Python constructor in `framelaw-py/src/lib.rs`
- [ ] `JsonCheck` round-trip
- [ ] Stub, export, docs, and tests on both sides

## Breaking change

<!-- Describe any breaking change and its migration path, or write "None". -->

## Notes for reviewers

<!-- Optional context, benchmarks, or follow-ups. -->

# Test Data

Test data is stored in `tests-data/` at the workspace root (not in the source
tree; gitignored). This binding-neutral location is shared by every Rust crate
and by the Python / future C / WASM bindings, and survives `cargo clean`.

Rust IO tests resolve it through a small local `common` helper in the io test
target (`molrs-io/tests/io/common.rs`), which just reads `../tests-data`; Python
tests resolve it through the `tests_data_dir` fixture in
`molrs-python/tests/conftest.py`. Both honor the `MOLRS_TESTS_DATA` environment
variable as an override.

## Fetch test data

```bash
bash scripts/fetch-test-data.sh
```

## Run tests

```bash
cargo test
```

## CI

Test data is cached in CI using `actions/cache@v4` with key based on the fetch script hash.

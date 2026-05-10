# Contributing to Documentation

The documentation system has one central invariant: public API prose starts in
Rust `///` comments. Zensical pages may explain workflows and concepts, but
reference pages should inject generated API docs or link to generated API docs
instead of copying signatures by hand.

For Python, keep `molrs-python/python/molrs/molrs.pyi` synchronized with the
PyO3 module exports. The freshness guard checks that every class and function
registered in `molrs-python/src/lib.rs` is declared in the stub.

For WASM, build declarations with the same `wasm-pack` flags used by npm
publishing. The generated `pkg/` directory is ignored and must not be committed.

Local documentation loop:

```bash
python scripts/check-python-stub-exports.py
maturin develop -m molrs-python/Cargo.toml
zensical build -f docs/zensical.toml
zensical serve -f docs/zensical.toml -a localhost:8000
```

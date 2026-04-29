# Installation

molrs is published as separate packages for Rust, Python, and npm. The packages
share the same core model, but each one follows the conventions of its host
ecosystem.

| Environment | Package | Command |
| --- | --- | --- |
| Rust | `molcrafts-molrs` | `cargo add molcrafts-molrs --features full` |
| Python | `molcrafts-molrs` | `python -m pip install molcrafts-molrs` |
| JavaScript / TypeScript | `@molcrafts/molrs` | `npm install @molcrafts/molrs` |

The Python import name is `molrs`. The npm package uses the scoped name
`@molcrafts/molrs`, while the generated TypeScript module exports classes such
as `Frame`, `Block`, `Box`, and analysis helpers directly.

## Verify the Environment

=== "Python"

    ```bash
    python -m pip install molcrafts-molrs
    python - <<'PY'
    import molrs

    ir = molrs.parse_smiles("O")
    print("components:", ir.n_components)
    print("atoms:", ir.to_atomistic().n_atoms)
    PY
    ```

    The import name is intentionally shorter than the package name. If
    `import molrs` fails after installation, check that the interpreter running
    the script is the same interpreter used by `python -m pip`.

=== "Rust"

    ```bash
    cargo new molrs-smoke
    cd molrs-smoke
    cargo add molcrafts-molrs --features full
    ```

    In `src/main.rs`:

    ```rust
    use molrs::smiles::{parse_smiles, to_atomistic};

    fn main() -> Result<(), Box<dyn std::error::Error>> {
        let ir = parse_smiles("O")?;
        let mol = to_atomistic(&ir)?;
        println!("atoms: {}", mol.n_atoms());
        Ok(())
    }
    ```

=== "TypeScript"

    ```bash
    npm install @molcrafts/molrs
    ```

    In an ESM-aware runtime or bundler:

    ```ts
    import init, { parseSMILES } from "@molcrafts/molrs";

    await init();
    console.log(parseSMILES("O").nComponents);
    ```

## Source Builds

For local Python development, install the extension module in editable form:

```bash
maturin develop -m molrs-python/Cargo.toml
```

For the npm package, build the same bundler target used by release publishing:

```bash
cd molrs-wasm
wasm-pack build --release --target bundler --scope molcrafts --out-name molrs
```

For documentation work, build the local site after the Python extension is
installed:

```bash
zensical build -f docs/zensical.toml
```

## Version Boundaries

Rust, Python, and npm packages are released separately but generated from the
same repository. If examples behave differently across languages, first check
the package versions. The documentation site follows the repository `master`
branch, while crates.io, PyPI, npm, and docs.rs describe released artifacts.

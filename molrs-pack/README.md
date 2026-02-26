# molrs-pack

`molrs-pack` is a Packmol-aligned Rust crate focused on behavior/performance
parity for 5 production-size examples.

## Run Examples

```bash
cargo run -p molrs-pack --release --example pack_mixture
cargo run -p molrs-pack --release --example pack_bilayer
cargo run -p molrs-pack --release --example pack_interface
cargo run -p molrs-pack --release --example pack_solvprotein
cargo run -p molrs-pack --release --example pack_spherical
```

## Batch Validation Test (all 5 examples)

```bash
cargo test -p molrs-pack --test examples_batch -- --ignored
```

## Packmol vs molrs-pack Time + Violation Metrics

```bash
cargo run -p molrs-pack --release --bin compare_examples
```

The command prints:
- time comparison table (`packmol`, `molrs-pack`, `ratio`)
- quantified violation metrics for both tools

To run a subset during smoke checks:

```bash
molrs-pack_CASES=mixture cargo run -p molrs-pack --release --bin compare_examples
```

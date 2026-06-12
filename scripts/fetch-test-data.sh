#!/usr/bin/env bash
set -euo pipefail

# Shared, binding-neutral test data: cloned to the workspace root so every
# crate (and the Python / future C / WASM bindings) resolves the same copy,
# and `cargo clean` no longer wipes it.
REPO_URL="https://github.com/MolCrafts/tests-data.git"
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TARGET_DIR="${MOLRS_TESTS_DATA:-$PROJECT_ROOT/tests-data}"

echo "Fetching test data to $TARGET_DIR..."

mkdir -p "$(dirname "$TARGET_DIR")"
rm -rf "$TARGET_DIR"
git clone --depth=1 "$REPO_URL" "$TARGET_DIR"

echo "Done. Run: cargo test"

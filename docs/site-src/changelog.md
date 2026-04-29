# Changelog

This page summarizes recent repository history for documentation readers. The
authoritative release history remains the Git tags and GitHub releases.

## 0.0.15

Version 0.0.15 continued the unified packaging work, including publish workflow
cleanup and formatting across the I/O crate.

## 0.0.12

The release workflow was hardened for WebAssembly and facade publishing. Python
publishing now passes interpreter discovery through the maturin container path.

## 0.0.11

Compute internals moved behind a unified compute DAG, and Python and WASM
bindings were adapted to the newer analysis shape.

## 0.0.10

RDF behavior was aligned more closely with freud-style normalization, and PCA
plus k-means analysis reached the public surface.

## 0.0.8

The facade crate was introduced so Rust users can depend on one public package
and opt into subsystems through Cargo features.

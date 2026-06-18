---
slug: compute-fit-04-dielectric
criteria:
  - id: ac-001
    summary: ε(ω) transform Fit reproduces legacy dielectric spectrum bit-for-bit
    type: code
    evaluator_hint: ""
    pass_when: |
      Given the raw dipole/current ACF that einstein_helfand_spectrum /
      green_kubo_spectrum compute internally, the new ε(ω) Fit applied to that ACF
      with the same window/FFT/metadata returns frequencies, eps_real, eps_imag
      arrays equal (== or documented float tol) to the legacy free-fn output.
    status: pending
  - id: ac-002
    summary: Legacy dielectric-spectrum free fns removed; transform lives in compute::fit
    type: code
    evaluator_hint: ""
    pass_when: |
      einstein_helfand_spectrum and green_kubo_spectrum no longer exist as free
      functions in compute/dielectric.rs; the ε(ω) transform is a Fit in
      compute::fit composing molrs::signal windows; windowed_acf_spectrum /
      windowed_acf_derivative_spectrum / dielectric acf_to_spectrum are removed
      (migrated) from dielectric.rs.
    status: pending
  - id: ac-003
    summary: Deprecated dielectric-spectrum PyO3 bindings + shims removed
    type: code
    evaluator_hint: ""
    pass_when: |
      dielectric_einstein_helfand_spectrum and dielectric_green_kubo_spectrum
      bindings are gone, registration converged; molpy DielectricSusceptibility
      uses raw compute + ε(ω) Fit and no longer triggers DeprecationWarning.
    status: pending
  - id: ac-004
    summary: Grep-clean of removed dielectric-spectrum symbols
    type: runtime
    evaluator_hint: "ripgrep over molrs/ molrs-python/ molpy/"
    pass_when: |
      grep finds zero live-code references to einstein_helfand_spectrum,
      green_kubo_spectrum, windowed_acf_spectrum, windowed_acf_derivative_spectrum,
      or dielectric::acf_to_spectrum across molrs, molrs-python, molpy.
    status: pending
  - id: ac-005
    summary: Compute gate green + molpy compute tests green on rebuilt wheel
    type: runtime
    evaluator_hint: ""
    pass_when: |
      cargo test -p molcrafts-molrs --features compute passes; clippy/fmt clean on
      touched files; maturin wheel rebuilt; pytest molpy/tests/test_compute
      -m "not external" passes with zero failures (no remaining dielectric-spectrum
      DeprecationWarnings).
    status: pending
  - id: ac-006
    summary: Dielectric physics unchanged (static limit + loss finite)
    type: scientific
    evaluator_hint: "marker: dielectric regression"
    pass_when: |
      ε(ω→0) recovers the Neumann static dielectric constant within prior tol; the
      loss spectrum ε″ stays finite to Nyquist (derivative-FT path); molpy
      DielectricSusceptibility output matches pre-migration within tolerance.
    status: pending
  - id: ac-007
    summary: Breaking version aligned across crate + bindings
    type: docs
    evaluator_hint: ""
    pass_when: |
      molcrafts-molrs is 0.2.0 and molrs-python pyproject.toml + molrs-ffi version
      are aligned to the 0.2.0 breaking release (no publish/tag performed).
    status: pending
---

# Acceptance criteria

- **ac-001 / ac-002** — the ε(ω) raw+fit kernel exists, reproduces the legacy output bit-for-bit, and the legacy free-fns/helpers are removed (chain fully closed: every compute returns raw).
- **ac-003 / ac-004** — bindings/shims purged; grep-clean of the last legacy symbols.
- **ac-005** — compute + molpy gates green on the rebuilt wheel.
- **ac-006** — dielectric physics unchanged (static limit + finite loss).
- **ac-007** — the 0.2.0 breaking version is consistent across the crate and its bindings.

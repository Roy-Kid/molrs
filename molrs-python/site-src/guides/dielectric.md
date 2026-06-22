# Dielectric Kernels

Technical reference for the dielectric kernels in `molrs-compute`: the
frequency-dependent permittivity $\varepsilon^*(\omega)$, the static dielectric
constant $\varepsilon(0)$, and the dipole / current builders they consume. Like
the [transport](transport.md) kernels these are array-based — the caller assembles
the per-frame collective quantities and the Rust layer does the correlation,
windowing, and transform.

This page is an **engineering reference**: signatures, return shapes, units, and
the compute→fit data flow. For the physics — linear-response theory, the
fluctuation–dissipation derivation, and worked from-principles examples — see the
MolPy guide [Dielectric Spectroscopy](https://molcrafts.github.io/molpy/compute/dielectric/).

## Architecture: raw compute vs fit

Every spectrum is a two-stage composition, and the split is deliberate:

- A **raw compute** measures an observable from the trajectory — an
  autocorrelation function, a collective MSD — and nothing else. It never windows,
  never FFTs, never fits.
- A **fit** consumes that raw observable plus the physical metadata and applies the
  window + transform + prefactor (or a regression) to produce the derived quantity.

So the FFT and the $1/3Vk_BT$ prefactors live in exactly one place (the `*Spectrum`
fits), and you can re-window or re-scale a cached ACF without re-reading the
trajectory.

| stage | Einstein–Helfand route | Green–Kubo route |
|-------|------------------------|------------------|
| **raw compute** | `DebyeRelaxation(V, T, bc).compute(M_D, dt, max_lag)` | `GreenKuboConductivity().compute(J, dt, max_lag)` |
| → returns | `{lag_times, acf, zero_lag_variance, volume, temperature, boundary}` | `{lag_times, jacf}` |
| **fit → ε(ω)** | `EinsteinHelfandSpectrum(dt, V, T, ε∞, zero_lag_variance).fit(acf)` | `GreenKuboSpectrum(dt, V, T, ε∞, window).fit(jacf)` |
| → returns | `{frequencies, eps_real, eps_imag}` | `{frequencies, eps_real, eps_imag}` |

Raw computes and fits are registered at the top level (`molrs.DebyeRelaxation`,
`molrs.EinsteinHelfandSpectrum`, …); the dipole/current/static-ε builders live
under `molrs.dielectric.Dielectric`.

## Why two routes

The collective dipole $\mathbf{M}(t)=\sum_i q_i\mathbf{r}_i$ of an electrolyte
splits into a bounded (rotational) part and a diffusive (conductive) part:

$$
\mathbf{M} = \underbrace{\textstyle\sum_{\text{solvent}} q_i\mathbf{r}_i}_{\mathbf{M}_D}
           + \underbrace{\textstyle\sum_{\text{ions}} q_i\mathbf{r}_i}_{\mathbf{M}_J}.
$$

$\mathbf{M}_D$ is bounded, so its ACF decays and transforms cleanly (Einstein–
Helfand). $\mathbf{M}_J$ is the integral of the ionic current and grows
diffusively, so the bare whole-dipole ACF diverges — the ions must enter through
the current $\mathbf{J}=\dot{\mathbf{M}}$ (Green–Kubo). Pick the route by what the
target contains, not by preference:

| route | observable | use for | low-$\omega$ |
|-------|-----------|---------|--------------|
| Einstein–Helfand | bounded dipole ACF | solvent / non-conducting | plateaus → $\varepsilon(0)$ |
| Green–Kubo | current ACF | any system incl. conduction | rises as $\sigma/\varepsilon_0\omega$ |

## Route A — `DebyeRelaxation` + `EinsteinHelfandSpectrum`

```python
import molrs

# M_D: (n_frames, 3) bounded dipole [e·Å]; dt [ps]; V [Å³]; T [K]; max_lag [frames]
raw  = molrs.DebyeRelaxation(V, T, "tinfoil").compute(M_D, dt, max_lag)
spec = molrs.EinsteinHelfandSpectrum(
    dt, V, T, epsilon_inf=1.0, zero_lag_variance=raw["zero_lag_variance"]
).fit(raw["acf"])
omega, eps_real, eps_loss = spec["frequencies"], spec["eps_real"], spec["eps_imag"]
```

- `DebyeRelaxation(volume, temperature, boundary="tinfoil")` — `boundary` is
  `"tinfoil"` (conducting) or `"vacuum"`. `.compute(dipoles, dt, max_lag)` returns
  the **unnormalized** ACF (`acf[0] == zero_lag_variance`) plus the metadata the
  amplitude needs.
- `EinsteinHelfandSpectrum(dt, volume, temperature, epsilon_inf, zero_lag_variance)`
  applies a one-sided $\cos^2$ taper + derivative-FT + the $4\pi\kappa/(3Vk_BT)$
  prefactor. `frequencies` are in rad·ps⁻¹.
- The relaxation time $\tau$ (Debye shape) is a separate fit: `molrs.DebyeFit` on
  the **normalized** ACF.

## Route B — `GreenKuboConductivity` + `GreenKuboSpectrum`

```python
# J: (n_frames, 3) current — finite-difference dM/dt or microscopic Σ q_i v_i
raw  = molrs.GreenKuboConductivity().compute(J, dt, max_lag)        # {lag_times, jacf}
spec = molrs.GreenKuboSpectrum(dt, V, T, epsilon_inf=1.0, window_type="hann").fit(raw["jacf"])
omega, eps_real, eps_loss = spec["frequencies"], spec["eps_real"], spec["eps_imag"]

# DC conductivity: integrate the same ACF (a fit, then a caller-applied prefactor)
run      = molrs.RunningIntegral().fit(raw["jacf"], dt)
sigma_dc = run["integral"][-1] / (3.0 * V * K_B * T)
```

- `GreenKuboSpectrum(dt, volume, temperature, epsilon_inf, window_type="hann")` —
  `window_type` is `"cosine_sq"`, `"hann"`, or `"blackman"`. It forms
  $\sigma(\omega)=\frac{1}{3Vk_BT}\!\int\!\langle\mathbf{J}(0)\mathbf{J}(t)\rangle e^{-i\omega t}dt$
  then $\varepsilon^*=\varepsilon_\infty+\sigma(\omega)/(i\omega\varepsilon_0)$.
- This is the **only** valid route for the conducting whole system. Its
  low-frequency $\varepsilon'$ converges more slowly than EH — budget a long
  trajectory and replicate averaging.
- The two current constructions ($\dot{\mathbf{M}}$ and $\sum q_i\mathbf{v}_i$) are
  equivalent; use whichever the trajectory provides.

## Builders — `molrs.dielectric.Dielectric`

Raw per-frame / per-trajectory kernels (all static methods):

| method | signature | returns |
|--------|-----------|---------|
| `compute_dipole_moment` | `(charges (n,), positions (n,3))` | $\mathbf{M}=\sum q_i\mathbf{r}_i$, `(3,)` |
| `compute_current_density` | `(dipoles (n_frames,3), dt, volume)` | current density `(n_frames,3)` |
| `decompose_current` | `(per_particle_current (n_frames,n_atoms,3), water_mask (n_atoms,) bool)` | `(J_water, J_ion)` each `(n_frames,3)` |
| `static_dielectric_constant` | `(dipoles (n_frames,3), volume, temperature, epsilon_inf)` | $\varepsilon(0)$ scalar |

`static_dielectric_constant` is the Neumann fluctuation formula
$\varepsilon(0)=\varepsilon_\infty+\frac{4\pi}{3Vk_BT}(\langle M^2\rangle-\langle M\rangle^2)$
— a pure kernel, no fit. `decompose_current` splits one per-particle current pass
into the solvent and ion collective currents (Route A's $\mathbf{M}_D$ derivative
and Route B's $\mathbf{J}$).

## Units & conventions

LAMMPS *real* units throughout:

| quantity | unit |
|----------|------|
| length | Å |
| charge | $e$ |
| time / `dt` | ps |
| temperature | K |
| volume | Å³ |
| dipole moment $\mathbf{M}$ | $e\cdot$Å |
| current density | $e\cdot$Å$^{-2}\cdot$ps$^{-1}$ |
| angular frequency $\omega$ | rad·ps⁻¹ |
| permittivity $\varepsilon$ | dimensionless |

- `max_lag` / `max_correlation_time` is in **frames** (clamped to
  $N_\text{frames}-1$); longer lags resolve lower frequencies but are noisier.
- `frequencies` are angular ω in rad·ps⁻¹ (divide by $2\pi$ for THz).
- Spectrum dicts expose `eps_imag` as the loss $\varepsilon''$ (positive);
  susceptibility is $\chi = \varepsilon - 1$.

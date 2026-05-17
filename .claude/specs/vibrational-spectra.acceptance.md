---
slug: vibrational-spectra
criteria:
  - id: ac-001
    summary: CosineSq window matches analytical formula
    type: code
    evaluator_hint: ""
    pass_when: |
      For N >= 2, apply_window with WindowType::CosineSq on an ArrayD of ones of length N along axis 0 returns an array where for every n in [0, N-1], the value equals cos²(π n / (2 (N-1))) to within 1e-12, and boundary values w[0] = 1.0, w[N-1] = 0.0 for N > 1.
    status: verified
    last_checked: 2026-05-17
  - id: ac-002
    summary: power_spectrum returns Spectrum with correct shapes
    type: code
    evaluator_hint: ""
    pass_when: |
      Given velocities of shape (100, 15) (100 frames, 5 atoms * 3), dt_fs=1.0, resolution=10, power_spectrum returns Ok(Spectrum) where frequencies_cm1.len() equals intensities.len() equals (4 * (resolution+1)).next_power_of_two() / 2 + 1, and n_frames == 100, resolution == 10.
    status: verified
    last_checked: 2026-05-17
  - id: ac-003
    summary: VDOS peak position matches known oscillation frequency
    type: scientific
    evaluator_hint: ""
    pass_when: |
      Given velocities with v_x(t) = sin(2π · 100 THz · t) where t = n · dt_fs for n = 0..255 and dt_fs = 0.5 fs, the peak in the power_spectrum output frequencies_cm1 (the index of the maximum intensity) is within 2 cm⁻¹ of 3335.64 cm⁻¹ (100 THz / (c·10¹⁰) approx 3335.64 cm⁻¹).
    status: verified
    last_checked: 2026-05-17
  - id: ac-004
    summary: ir_spectrum rejects invalid inputs
    type: code
    evaluator_hint: ""
    pass_when: |
      ir_spectrum with dipole_moments.shape=(1, 3) returns Err(ComputeError::EmptyInput); ir_spectrum with dt_fs <= 0 returns Err(ComputeError::OutOfRange); ir_spectrum with dipole_moments.shape=(10, 4) returns Err(ComputeError::DimensionMismatch).
    status: verified
    last_checked: 2026-05-17
  - id: ac-005
    summary: IR spectrum of oscillating dipole produces peak at correct frequency
    type: scientific
    evaluator_hint: ""
    pass_when: |
      Given dipole_moments with M_z(t) = sin(2π · 50 THz · t), t = n · 0.5 fs for n = 0..511, ir_spectrum returns a spectrum whose maximum intensity frequency is within 2 cm⁻¹ of 1667.82 cm⁻¹ (50 THz to cm⁻¹ conversion).
    status: verified
    last_checked: 2026-05-17
  - id: ac-006
    summary: raman_spectrum returns RamanSpectrum with correct fields
    type: code
    evaluator_hint: ""
    pass_when: |
      Given polarizabilities of shape (100, 6), dt_fs=1.0, resolution=10, incident_frequency_cm1=0.0, temperature_k=300.0, averaged=false, raman_spectrum returns Ok(RamanSpectrum) where frequencies_cm1.len() == isotropic.len() == anisotropic.len(), parallel.is_none(), perpendicular.is_none(), n_frames == 100.
    status: verified
    last_checked: 2026-05-17
  - id: ac-007
    summary: Raman with averaged=true produces parallel and perpendicular
    type: code
    evaluator_hint: ""
    pass_when: |
      Given polarizabilities of shape (100, 6), dt_fs=1.0, resolution=10, incident_frequency_cm1=0.0, temperature_k=0.0, averaged=true, raman_spectrum returns Ok(RamanSpectrum) where parallel.is_some() and perpendicular.is_some(), and parallel.as_ref().unwrap().len() == perpendicular.as_ref().unwrap().len() == frequencies_cm1.len().
    status: verified
    last_checked: 2026-05-17
  - id: ac-008
    summary: Raman isotropic/anisotropic intensities are non-zero for varying signal
    type: runtime
    evaluator_hint: ""
    pass_when: |
      Given polarizabilities of shape (256, 6) where the Voigt components vary sinusoidally in time, both isotropic and anisotropic RMS intensities are > 0.1 (arbitrary units, saturating a non-zero signal sanity check).
    status: verified
    last_checked: 2026-05-17
  - id: ac-009
    summary: Frequency conversion from rad/fs to cm⁻¹ is numerically exact
    type: scientific
    evaluator_hint: ""
    pass_when: |
      The conversion factor visible in the output frequencies_cm1 for a known signal matches ω · 33356.4 where ω is in rad/fs, to within 0.01 cm⁻¹ at the first non-DC frequency bin.
    status: verified
    last_checked: 2026-05-17
  - id: ac-010
    summary: CosineSq variant accessible via WindowType
    type: code
    evaluator_hint: ""
    pass_when: |
      molrs_signal::WindowType::CosineSq compiles, and match on the variant reaches the new branch; applying CosineSq to a random 1D signal of length N returns an array with the same shape and all elements finite.
    status: verified
    last_checked: 2026-05-17
---

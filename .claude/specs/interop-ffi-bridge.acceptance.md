---
slug: interop-ffi-bridge
criteria:
  - id: ac-001
    summary: A molrs.Frame capsule resolves to a shared FrameRef (zero copy)
    type: code
    evaluator_hint: "Build a molrs.Frame in Python, pass to a Rust consumer that links molrs-ffi."
    pass_when: |
      molpack::interop::frame_from_py(obj) resolves a "molrs.FrameRef" capsule to a
      molrs_ffi::FrameRef backed by the SAME store as the Python frame (Rc clone, no deep
      copy); FrameRef::with_frame/with_frame_mut lends the real core &molrs::Frame so the
      consumer calls the full core API directly. Works on molrs.Frame and molpy.Frame.
    status: met
    last_checked: "2026-06-13"
  - id: ac-002
    summary: ForceField crosses the boundary via the same capsule path
    type: code
    evaluator_hint: ""
    pass_when: |
      molrs.ForceField exposes _ffi_forcefield_capsule(); molrs-ffi gains ForceFieldRef +
      with_forcefield; molpack::interop::forcefield_from_py yields a handle that lends the
      core &ForceField. No to_dict/from_dict. (Consumption by LBFGSRelaxer(ff) + special_bonds
      reading are in the sibling spec ff-special-bonds-nblist; the resolver compiles + is wired.)
    status: met
    last_checked: "2026-06-13"
  - id: ac-003
    summary: Return path — a consumer-built frame becomes a Python molrs.Frame
    type: code
    evaluator_hint: ""
    pass_when: |
      molpack::interop::frame_to_py(py, &frame, box) returns an isinstance-correct rich
      molrs.Frame via Frame._from_ffi_frameref_capsule (Rust base resolver + Python-layer
      upgrade), with no column marshalling. Verified: result.frame is molrs.Frame, coords/
      elements/periodic box all correct.
    status: met
    last_checked: "2026-06-13"
  - id: ac-004
    summary: Data contract — uint atom indices end to end
    type: code
    evaluator_hint: ""
    pass_when: |
      Atom-connectivity index columns (atomi/atomj/atomk/atoml, ids) stay UInt (molrs U=u32)
      across the capsule path; the marshalling-era int32/UInt mismatch cannot recur. Verified:
      a uint32 atomi column survives capsule export -> _from_ffi_frameref_capsule round-trip as
      uint32 (no signedness collapse). The molpy-reader uint sweep is tracked separately.
    status: met
    last_checked: "2026-06-13"
  - id: ac-005
    summary: Version guard on the capsule ABI
    type: code
    evaluator_hint: ""
    pass_when: |
      DEFERRED (revised design, one consumer). Capsule names stay "molrs.FrameRef" /
      "molrs.ForceFieldRef" (no .vN tag yet); both wheels pin the same molcrafts-molrs-ffi
      0.1.1 manually (path+version), which is the soundness contract. Add an ABI tag when a
      second independent consumer appears.
    status: deferred
    last_checked: "2026-06-13"
  - id: ac-006
    summary: molpack reference migration drops frame_marshal
    type: code
    evaluator_hint: "molpack is the reference consumer."
    pass_when: |
      molpack/python/src/frame_marshal.rs is deleted; molpack-python links molcrafts-molrs-ffi;
      Target, PackResult, and the relaxer bindings use molpack::interop; build + a pack run
      succeed with zero to_dict/from_dict on the molpack <-> molrs boundary. Verified: 148
      molpack pytest pass; plain-dict frame input dropped (per locked decision — molpack takes
      only molrs/molpy Frames).
    status: met
    last_checked: "2026-06-13"
---

---
slug: frame-block-subclass
criteria:
  - id: ac-001
    summary: PyFrame pyclass attribute includes subclass
    type: code
    evaluator_hint: ""
    pass_when: |
      grep -qE 'pyclass\(name = "Frame".*subclass' molrs-python/src/frame.rs
    status: verified
    last_checked: 2026-05-28

  - id: ac-002
    summary: PyBlock pyclass attribute includes subclass
    type: code
    evaluator_hint: ""
    pass_when: |
      grep -qE 'pyclass\(name = "Block".*subclass' molrs-python/src/block.rs
    status: verified
    last_checked: 2026-05-28

  - id: ac-003
    summary: Python can subclass molrs.Frame
    type: code
    evaluator_hint: ""
    pass_when: |
      cd molrs-python && maturin develop --release && python -c "
      import molrs
      class Sub(molrs.Frame): pass
      s = Sub()
      assert isinstance(s, molrs.Frame)
      "
    status: verified
    last_checked: 2026-05-28

  - id: ac-004
    summary: Python can subclass molrs.Block
    type: code
    evaluator_hint: ""
    pass_when: |
      cd molrs-python && python -c "
      import molrs
      class Sub(molrs.Block): pass
      s = Sub()
      assert isinstance(s, molrs.Block)
      "
    status: verified
    last_checked: 2026-05-28

  - id: ac-005
    summary: subclass instances accepted by molrs APIs expecting Frame / Block
    type: code
    evaluator_hint: ""
    pass_when: |
      cd molrs-python && pytest tests/test_subclass_frame.py -v
    status: verified
    last_checked: 2026-05-28

  - id: ac-006
    summary: clippy + fmt clean
    type: code
    evaluator_hint: ""
    pass_when: |
      cargo clippy -p molrs-python --all-targets -- -D warnings
      cargo fmt --check
    status: verified
    last_checked: 2026-05-28
---

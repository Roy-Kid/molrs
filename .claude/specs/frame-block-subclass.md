---
title: Frame / Block Python-subclassable
status: code-complete
created: 2026-05-28
---

# Frame / Block Python-subclassable

## Summary

为 `molrs.Frame` 与 `molrs.Block` 的 PyO3 `#[pyclass]` 注解补上 `subclass`
标志，使 Python 端可以 `class MyFrame(molrs.Frame): ...`。这是
`molpy.Box ← molrs.Box` 已经验证过的同一模式（参见 molpy spec
`molrs-backend` Phase 0），现在向 Frame / Block 推广，让 molpy 把
`molpy.Frame` / `molpy.Block` 从「组合 + `_inner`」改成「真子类」,
从而能被任何 `molrs.*` API 直接接收，无需 `.to_molrs()` 桥接。

零行为变化：被继承的子类拿到的依旧是同一个 Rust slot，方法分发不变。

## Domain basis

无新物理；纯绑定层修改。

PyO3 `#[pyclass]` 默认是 final（`type.__new__` 拒绝 Python 子类化）。
要允许子类必须显式加 `subclass` 属性，PyO3 会在生成的类型对象里把
`Py_TPFLAGS_BASETYPE` 打开。`from_py_object` 不变 ——
`FromPyObject::extract` 走 downcast 到本类型，子类实例满足 `isinstance`
检查，因此既有签名 `fn f(arg: PyRef<PyFrame>)` 仍然接受
`molpy.Frame` 实例。

参考：PyO3 user guide §Inheritance；同仓库 `simbox.rs:35` 已是此写法。

## Design

### 改动

仅两行注解：

```rust
// molrs-python/src/frame.rs:56
- #[pyclass(name = "Frame", from_py_object, unsendable)]
+ #[pyclass(name = "Frame", from_py_object, unsendable, subclass)]

// molrs-python/src/block.rs:83
- #[pyclass(name = "Block", from_py_object, unsendable)]
+ #[pyclass(name = "Block", from_py_object, unsendable, subclass)]
```

### 验证

新增 `molrs-python/tests/test_subclass_frame.py`，覆盖：

1. `class SubFrame(molrs.Frame): pass; SubFrame()` 不抛 `TypeError`
2. 子类实例 `isinstance(sub, molrs.Frame)` 为真
3. 父类方法（`__setitem__` / `keys` / `box`）继承可用
4. 子类实例传入任意接受 `molrs.Frame` 的 PyO3 API（例如
   `molrs.write_xyz(path, sub_frame)`），不报类型错误
5. 同上 4 条对 `Block` 重复一次

### 与 `from_py_object` 的关系

`from_py_object` 走 downcast 到本类。子类实例命中 `isinstance` 检查，
downcast 成功，拿到的是子类附带的父类 slot —— 子类在 Python 侧多挂
的 `__dict__` 属性在 Rust 端**不可见**，这是 by-design：molrs kernel
只关心 Rust slot，Python 衍生状态由 Python 自己管。

### 不引入的东西

- 不动 `unsendable`；Frame / Block 内含 `Rc`，仍然是单线程。
- 不暴露 `subclass` 给 wasm / capi —— 这条仅对 PyO3 绑定。
- 不动 `FrameRef` / `BlockRef` 的 Rust 接口。
- 不发新 minor；这是补丁级。

## Files to create or modify

- `molrs-python/src/frame.rs` — (modify) 第 56 行 pyclass 属性添加 `subclass`
- `molrs-python/src/block.rs` — (modify) 第 83 行 pyclass 属性添加 `subclass`
- `molrs-python/tests/test_subclass_frame.py` — (new) Frame + Block 子类化烟雾测试

## Tasks

- [x] 在 `frame.rs:56` 给 `#[pyclass]` 添加 `subclass` 属性
- [x] 在 `block.rs:83` 给 `#[pyclass]` 添加 `subclass` 属性
- [x] 写 `tests/test_subclass_frame.py`，覆盖 5 条验证项（见 Design §验证）
- [x] `maturin develop --release`，跑 `pytest tests/test_subclass_frame.py -v`
- [x] `cargo clippy -p molrs-python -- -D warnings`、`cargo fmt --check`

## Testing strategy

### Happy path

子类实例化、`isinstance` 检查、继承方法调用、传入 molrs API 路径
均通过（见 Design §验证）。

### 边界

- `SubFrame.__init__` 调用 `super().__init__()` 后能正常
  `frame["atoms"] = Block()` —— 确认子类没有打破 Rust slot 初始化。
- 把子类实例**两次**传给同一个 molrs API 调用，第二次仍然成功 ——
  确认 `from_py_object` 提取不消费实例。

### 不在测试中

- 性能基准 —— 注解修改不引入新代码路径。
- 跨进程序列化、`__reduce__` —— 与 subclassability 无关。

## Out of scope

- molpy 侧的继承重构：独立 spec `frame-block-inherit-molrs`
  （在 molpy repo），本 spec 只交付 molrs 端的前置条件。
- 把 `Frame.metadata` / `Block` object-dtype 列下沉到 Rust 端 ——
  独立优化，与本 spec 正交。
- 释放新 PyPI wheel —— 本地 `maturin develop` 已足够联调。

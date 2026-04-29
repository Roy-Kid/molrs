#!/usr/bin/env python3
"""Check that every PyO3 module export is declared in molrs.pyi."""

from __future__ import annotations

import re
import sys
import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "molrs-python" / "src"
LIB_RS = SRC / "lib.rs"
STUB = ROOT / "molrs-python" / "python" / "molrs" / "molrs.pyi"
INIT = ROOT / "molrs-python" / "python" / "molrs" / "__init__.py"


def exported_classes(lib_text: str, class_names: dict[str, str]) -> set[str]:
    exports: set[str] = set()
    for item in re.findall(r"m\.add_class::<([^>]+)>", lib_text):
        rust_name = item.split("::")[-1].strip()
        exports.add(class_names.get(rust_name, rust_name))
    return exports


def exported_functions(lib_text: str, function_names: dict[str, str]) -> set[str]:
    exports: set[str] = set()
    for item in re.findall(r"wrap_pyfunction!\(([^,\)]+),\s*m\)", lib_text):
        rust_name = item.split("::")[-1].strip()
        exports.add(function_names.get(rust_name, rust_name))
    return exports


def collect_class_names() -> dict[str, str]:
    pattern = re.compile(
        r"#\[pyclass(?P<attrs>(?:\([^\]]*\))?)\]"
        r"(?P<extra>(?:\s*#\[[^\]]+\])*)"
        r"\s*pub\s+struct\s+(?P<rust>\w+)",
        re.MULTILINE,
    )
    names: dict[str, str] = {}
    for path in SRC.rglob("*.rs"):
        text = path.read_text(encoding="utf-8")
        for match in pattern.finditer(text):
            attrs = match.group("attrs") or ""
            py_name = re.search(r'name\s*=\s*"([^"]+)"', attrs)
            names[match.group("rust")] = py_name.group(1) if py_name else match.group("rust")
    return names


def collect_function_names() -> dict[str, str]:
    pattern = re.compile(
        r"#\[pyfunction(?:\([^\]]*\))?\]"
        r"(?P<extra>(?:\s*#\[[^\]]+\])*)"
        r"\s*pub\s+fn\s+(?P<rust>\w+)",
        re.MULTILINE,
    )
    names: dict[str, str] = {}
    for path in SRC.rglob("*.rs"):
        text = path.read_text(encoding="utf-8")
        for match in pattern.finditer(text):
            extra = match.group("extra") or ""
            py_name = re.search(r"#\[pyo3\([^\]]*name\s*=\s*\"([^\"]+)\"", extra)
            names[match.group("rust")] = py_name.group(1) if py_name else match.group("rust")
    return names


def declared_stub_symbols(stub_text: str) -> set[str]:
    return set(re.findall(r"^(?:class|def)\s+([A-Za-z_]\w*)\b", stub_text, re.MULTILINE))


def init_all_symbols(init_text: str) -> set[str]:
    tree = ast.parse(init_text)
    for node in tree.body:
        if isinstance(node, ast.Assign):
            if any(isinstance(target, ast.Name) and target.id == "__all__" for target in node.targets):
                if isinstance(node.value, (ast.List, ast.Tuple)):
                    return {
                        item.value
                        for item in node.value.elts
                        if isinstance(item, ast.Constant) and isinstance(item.value, str)
                    }
    return set()


def main() -> int:
    lib_text = LIB_RS.read_text(encoding="utf-8")
    stub_text = STUB.read_text(encoding="utf-8")
    init_text = INIT.read_text(encoding="utf-8")

    expected = exported_classes(lib_text, collect_class_names())
    expected |= exported_functions(lib_text, collect_function_names())
    declared = declared_stub_symbols(stub_text)
    init_exports = init_all_symbols(init_text)

    missing = sorted(expected - declared)
    missing_init = sorted(expected - init_exports)
    extra_context = sorted(declared - expected)

    if missing:
        print("molrs.pyi is missing PyO3 exports:")
        for name in missing:
            print(f"  - {name}")
        return 1
    if missing_init:
        print("molrs.__all__ is missing PyO3 exports:")
        for name in missing_init:
            print(f"  - {name}")
        return 1

    print(f"molrs.pyi and molrs.__all__ declare all {len(expected)} PyO3 exports from src/lib.rs.")
    if extra_context:
        print(f"Additional stub-only helper symbols: {len(extra_context)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

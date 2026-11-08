"""Architecture layering, enforced by an AST scan of every import (incl. function-local ones).

analytics/   pure maths: nothing from engines / convergence / ui / data (IO) / pipeline / streamlit
engines/     not from convergence / ui / pipeline / streamlit
convergence/ not from ui / pipeline / streamlit
data/        the IO boundary: not from ui / pipeline / streamlit
"""
import ast
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
RULES = {
    "analytics": {"engines", "convergence", "ui", "data", "pipeline", "app", "streamlit"},
    "engines": {"convergence", "ui", "pipeline", "app", "streamlit"},
    "convergence": {"ui", "pipeline", "app", "streamlit"},
    "data": {"ui", "pipeline", "app", "streamlit"},
}


def _imports(path: Path) -> set[str]:
    rel = path.relative_to(REPO).with_suffix("")
    return _imports_src(path.read_text(encoding="utf-8"), list(rel.parts[:-1]), str(path))


def _imports_src(src: str, pkg_parts: list[str], filename: str = "<src>") -> set[str]:
    mods = set()
    for node in ast.walk(ast.parse(src, filename=filename)):
        if isinstance(node, ast.Import):
            mods.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = pkg_parts[: len(pkg_parts) - (node.level - 1)] if node.level > 1 else pkg_parts
                mod = ".".join(base + ([node.module] if node.module else []))
            else:
                mod = node.module or ""
            mods.add(mod)
            if not node.module:                       # `from . import x` → x is a submodule
                mods.update(f"{mod}.{a.name}" for a in node.names)
    return {m for m in mods if m}


@pytest.mark.parametrize("pkg", sorted(RULES))
def test_layering(pkg):
    files = sorted((REPO / pkg).rglob("*.py"))
    assert files, f"{pkg}/ has no modules"
    bad = []
    for f in files:
        for m in _imports(f):
            if m.split(".")[0] in RULES[pkg]:
                bad.append(f"{f.relative_to(REPO)} imports {m}")
    assert bad == []


def test_scanner_sees_relative_and_local_imports():
    """Self-check of the scanner: function-local and relative imports are resolved."""
    src = "def f():\n    import streamlit as st\nfrom ..engines import aarambh\nfrom . import hurst\n"
    got = _imports_src(src, ["analytics", "sub"])
    assert "streamlit" in got and "analytics.engines" in got and "analytics.sub.hurst" in got
    assert {m.split(".")[0] for m in _imports_src("from .. import ui\n", ["data", "x"])} == {"data"}

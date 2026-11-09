"""
Nishkarsh — Convergence: do the systems agree? (≈ Tattva's Convergence page; lean words only.)

PLACEHOLDER (phase A): renders the page's first section header and an empty state. Phase B
fills it per DESIGN_SPEC_v4 §3 "Convergence": KPI strip ×6 → "The Read — Three Pillars,
Averaged" (3-row STACKED panel, window=True) → "What Drives Today's Read" (``drivers_rows``
below) → "Agreement Anatomy" → "Agreement Lift" → "Walk-Forward Edge"
(``pillar_edge["agreement"]``) → "Divergences" (amber; ``conclusion_events``). Windowed page.

``drivers_rows`` is the pure row builder for "What Drives Today's Read" (ported from the
retired Today page; tested in tests/test_ui_pages.py).
"""

from __future__ import annotations

import streamlit as st

from convergence.nishkarsh import COMPONENTS
from ui.components import (
    PILLARS, _d, _list, _str, num, render_empty_state, render_section_header,
)
from ui.copy import NKO_COMPONENT_PLAIN, NKO_PILLAR_PLAIN


def _component_label(cid, fallback="") -> str:
    return NKO_COMPONENT_PLAIN.get(_str(cid), (_str(fallback) or "An input", ""))[0]


def drivers_rows(k: dict) -> list[dict]:
    """Pillars and, under each, its inputs — today's z on one scale; the top-3 drivers marked.
    ``[{kind: pillar|input, label, z, driver, hint}]``."""
    k = _d(k)
    comps = {_str(_d(c).get("id")): _d(c) for c in _list(k.get("components"))}
    top = {_str(_d(c).get("id")) for c in _list(k.get("drivers"))}
    pil = _d(k.get("pillars"))
    rows = []
    for p in PILLARS:
        rows.append({"kind": "pillar", "label": NKO_PILLAR_PLAIN[p][0], "z": num(pil.get(p)), "driver": False,
                     "hint": NKO_PILLAR_PLAIN[p][1]})
        for pil_, cid, lab, _ in COMPONENTS:
            if pil_ == p:
                rows.append({"kind": "input", "label": _component_label(cid, lab),
                             "z": num(comps.get(cid, {}).get("z")), "driver": cid in top,
                             "hint": NKO_COMPONENT_PLAIN.get(cid, ("", ""))[1]})
    return rows


def render_convergence_page() -> None:
    v = _d(_d(st.session_state.get("conclusion")).get("verdict"))
    desc = _str(v.get("headline"), "Aarambh's top-down forecast and Nirnay's bottom-up breadth, scored against "
                "each other across direction, breadth, magnitude and regime.")
    render_section_header("Convergence — Do the Systems Agree?", desc, icon="target", accent="accent")
    render_empty_state("Being rebuilt", "This page is moving onto the new design system. The agreement pillar "
                       "is on Overview's Inference Chain meanwhile.", eyebrow="Convergence")

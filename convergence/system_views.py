"""
System page views — every engine's walk-forward edge in the ONE edge contract.

The read and its three pillars already carry Tattva's edge contract in the conclusion payload
(``nko["edge"]`` / ``nko["pillar_edge"]``, built by ``convergence.nishkarsh.edge_contract``).
Two more graded series sit in the payload as raw ``ic_evidence`` dicts — the signed agreement
oscillator (Edge Ledger L-12, ``oscillator["evidence"]``) and Aarambh's forecast (L-03,
``edge``). This re-shapes those two through the same ``edge_contract`` so the System page can
put all six side by side, graded one way.

Nothing is re-estimated: the IC, n and p are the payload's own; the periods are the calendar
years ``ic_evidence`` already scored (spanning the year, since the dated series are not in the
payload); the SE is ``edge_contract``'s own mapping of the same Fisher-z variance. A series the
payload itself calls ungraded (Aarambh's ``available`` False, the oscillator's INSUFFICIENT_DATA)
gets an empty contract, so the table and the Ledger use the same words. Pure — no Streamlit, no
I/O; malformed input degrades to an empty contract (``ic`` None).
"""

from __future__ import annotations

import math

import pandas as pd

from convergence.nishkarsh import edge_contract

__all__ = ["SYSTEM_EDGE_KEYS", "system_edges"]

#: The graded series, in reading order: the read (L-13), its three pillars, the signed agreement
#: oscillator (L-12) and Aarambh's forecast (L-03).
SYSTEM_EDGE_KEYS = ("read", "top_down", "bottom_up", "agreement", "oscillator", "aarambh")

_EMPTY = pd.Series(dtype=float)


def _d(x) -> dict:
    return x if isinstance(x, dict) else {}


def _finite(x) -> float | None:
    if x is None or isinstance(x, bool):
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _blank(h: int) -> dict:
    return {"ic": None, "se": None, "n": 0, "p": None, "horizon": int(h), "blocks": [],
            "positive_blocks": 0, "scored_blocks": 0}


def _clean_contract(e, h: int) -> dict:
    """A contract already in the payload, with anything non-finite or malformed dropped."""
    e = _d(e)
    if _finite(e.get("ic")) is None:
        return _blank(h)
    blocks = [{"start": str(b.get("start") or ""), "end": str(b.get("end") or ""), "ic": _finite(b.get("ic"))}
              for b in (e.get("blocks") if isinstance(e.get("blocks"), list) else []) if isinstance(b, dict)]
    blocks = [b for b in blocks if b["ic"] is not None]
    n, pos, scored = (_finite(e.get(x)) for x in ("n", "positive_blocks", "scored_blocks"))
    return {"ic": _finite(e.get("ic")), "se": _finite(e.get("se")), "n": int(n) if n is not None else 0,
            "p": _finite(e.get("p")), "horizon": int(_finite(e.get("horizon")) or h), "blocks": blocks,
            "positive_blocks": int(pos) if pos is not None else sum(b["ic"] > 0 for b in blocks),
            "scored_blocks": int(scored) if scored is not None else len(blocks)}


def _from_evidence(e, h: int) -> dict:
    """An ``ic_evidence`` dict → the edge contract (via ``edge_contract``; nothing re-estimated)."""
    e = _d(e)
    ic, n = _finite(e.get("ic")), _finite(e.get("n"))
    if ic is None or n is None or e.get("available") is False:
        return _blank(h)
    yearly = {str(y): _finite(x) for y, x in _d(e.get("yearly")).items()
              if str(y).isdigit() and _finite(x) is not None}
    return edge_contract({"ic": ic, "n": int(n), "p": _finite(e.get("p")), "yearly": yearly}, _EMPTY, _EMPTY, h)


def system_edges(conclusion) -> dict[str, dict]:
    """``{key: edge contract}`` for every key in ``SYSTEM_EDGE_KEYS`` — always all six keys; a
    series with no scored history gets an empty contract (``ic`` None, no blocks)."""
    con = _d(conclusion)
    h = int(_finite(con.get("horizon")) or 10)
    k = _d(con.get("nko"))
    pe = _d(k.get("pillar_edge"))
    return {
        "read": _clean_contract(k.get("edge"), h),
        "top_down": _clean_contract(pe.get("top_down"), h),
        "bottom_up": _clean_contract(pe.get("bottom_up"), h),
        "agreement": _clean_contract(pe.get("agreement"), h),
        "oscillator": (_blank(h) if str(_d(con.get("oscillator")).get("status") or "").upper() == "INSUFFICIENT_DATA"
                       else _from_evidence(_d(con.get("oscillator")).get("evidence"), h)),
        "aarambh": _from_evidence(con.get("edge"), h),
    }

"""Shared test setup.

Thread pins and the numba cache dir are set BEFORE numpy / numba are imported so the
suite is deterministic and does not oversubscribe the CPU (BLAS reductions are
order-dependent across thread counts).
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ.setdefault("NUMBA_CACHE_DIR", str(Path(tempfile.gettempdir()) / "nishkarsh-numba-cache"))
_SESSION_CACHE = Path(tempfile.mkdtemp(prefix="nishkarsh-test-cache-"))
os.environ["NISHKARSH_CACHE_DIR"] = str(_SESSION_CACHE)   # never touch ~/.cache/nishkarsh

REPO = Path(__file__).resolve().parents[1]
TESTS = Path(__file__).resolve().parent
for _p in (str(TESTS), str(REPO)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import logging  # noqa: E402

import numpy as np  # noqa: E402,F401
import pandas as pd  # noqa: E402,F401
import pytest  # noqa: E402

logging.getLogger().setLevel(logging.ERROR)


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO


@pytest.fixture(scope="session")
def small_sheet():
    """Seeded synthetic sheet (DATE ISO strings + NIFTY50_PE + 14 predictors), 900 rows."""
    from data.synthetic import synthetic_sheet
    return synthetic_sheet(7, n_rows=900)


@pytest.fixture(scope="session")
def small_world():
    """Seeded synthetic world → (sheet_df, SyntheticSources)."""
    from data.synthetic import synthetic_world
    return synthetic_world(7, n_rows=900)


@pytest.fixture
def cache_root(tmp_path):
    """Re-root every module-level data cache into a fresh tmp dir; restore afterwards."""
    from data import cache as DC
    root = tmp_path / "cache"
    DC.set_cache_root(root)
    saved_ttl = {id(c): c.ttl for c in DC.all_caches()}
    try:
        yield root
    finally:
        for c in DC.all_caches():
            c.ttl = saved_ttl[id(c)]
        DC.set_cache_root(_SESSION_CACHE)

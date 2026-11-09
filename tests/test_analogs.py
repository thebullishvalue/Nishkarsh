"""analytics/analogs.py core — OAS shrinkage (A4-13). The NKO-state matcher, its causality,
Theiler separation and base rate are tested in tests/test_nishkarsh.py."""
import numpy as np
import pytest

import analytics.analogs as AN


def test_oas_matches_sklearn():
    OAS = pytest.importorskip("sklearn.covariance").OAS
    rng = np.random.default_rng(0)
    for p, n in [(3, 40), (4, 200), (8, 60), (20, 30)]:
        for kind in ("random", "isotropic"):
            A = rng.standard_normal((p, p)) if kind == "random" else np.eye(p)
            X = rng.standard_normal((n, p)) @ A.T
            S = np.cov(X, rowvar=False, bias=True)
            ref = OAS().fit(X).covariance_
            np.testing.assert_allclose(AN._oas_shrinkage(S, n), ref, rtol=0, atol=1e-10, err_msg=f"{p}x{n} {kind}")


def test_oas_alias_kept():
    assert AN._ledoit_wolf_shrinkage is AN._oas_shrinkage

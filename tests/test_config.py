"""core/config.py coherence (validate_config runs at import; v1.6.0 shipped test-mode 15/30 windows)."""
import pytest

import core.config as cfg


def test_validate_config_passes():
    cfg.validate_config()


def test_min_train_covers_ten_p_plus_h():
    p = len(cfg.DEFAULT_PREDICTORS)
    assert cfg.MIN_TRAIN_SIZE >= 10 * p + cfg.AARAMBH_FWD_HORIZON
    assert cfg.MAX_TRAIN_SIZE >= cfg.MIN_TRAIN_SIZE


def test_conv2_and_edge_constants_sane():
    assert 0 < cfg.CONV2_TIER_WEAK < cfg.CONV2_TIER_MODERATE < cfg.CONV2_TIER_STRONG < 100
    assert cfg.BACKTEST_REFERENCE_BPS in cfg.BACKTEST_COST_BPS
    assert 1 <= cfg.AARAMBH_WF_REFIT <= cfg.AARAMBH_FWD_HORIZON
    assert cfg.CONV2_ZMIN_PERIODS == 126 and cfg.CONV2_ZCLIP == 4.0


@pytest.mark.parametrize("attr, value, needle", [
    ("MIN_TRAIN_SIZE", 15, "MIN_TRAIN_SIZE"),            # the v1.6.0 test-mode window
    ("MAX_TRAIN_SIZE", 100, "MAX_TRAIN_SIZE"),           # < MIN_TRAIN_SIZE
    ("AARAMBH_WF_REFIT", 0, "AARAMBH_WF_REFIT"),
    ("AARAMBH_WF_REFIT", 11, "AARAMBH_WF_REFIT"),        # > horizon
    ("CONV2_TIER_WEAK", 70, "tiers"),
    ("EDGE_ALPHA", 0.7, "EDGE_ALPHA"),
    ("BACKTEST_REFERENCE_BPS", 3.0, "BACKTEST_REFERENCE_BPS"),
    ("CALIBRATION_RETURN_LABEL", "bogus", "CALIBRATION_RETURN_LABEL"),
])
def test_broken_invariant_raises(monkeypatch, attr, value, needle):
    monkeypatch.setattr(cfg, attr, value)
    with pytest.raises(ValueError, match=needle):
        cfg.validate_config()

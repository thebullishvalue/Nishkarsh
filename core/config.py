"""
Nishkarsh v1.4.0 — Configuration constants, thresholds, column mappings, and shared defaults.
निष्कर्ष (Nishkarsha) — "Conclusion / Inference"

CORE — Merged from both Aarambh (correl.py) and Nirnay (nirnay_core.py) monoliths.
"""

# ─── Version / Product ───────────────────────────────────────────────────────

VERSION = "1.4.23"
PRODUCT_NAME = "NISHKARSH"
COMPANY = "@thebullishvalue"

# ─── Aarambh Engine Defaults ─────────────────────────────────────────────────

LOOKBACK_WINDOWS = (5, 10, 20, 50, 100)
# Walk-forward training window (trading days). The engine fits a ~14-feature
# regularized ensemble, so the window must be >> feature count to stay
# well-conditioned. MIN bootstraps the expanding window at one trading year;
# MAX caps it at three years so the model stays adaptive to regime drift.
# (Previously 15/30 — degenerate p/n for a 14-feature fit.)
MIN_TRAIN_SIZE = 1500
MAX_TRAIN_SIZE = 2000
REFIT_INTERVAL = 10
RIDGE_ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0)
HUBER_EPSILON = 1.35
# 200 is bit-identical to 500 here: on real data the walk-forward Huber fits
# converge well before the cap (0 of 140 chunks hit 150 iters), and OOS R²/IC
# are unchanged from max_iter 200→500. Trimmed purely to drop wasted headroom.
HUBER_MAX_ITER = 200
OU_PROJECTION_DAYS = 90
MIN_DATA_POINTS = 1500

# When True, Aarambh trains on causal-PCA factors of (sheet predictors + the
# full macro panel) instead of the raw sheet predictors. Combines everything,
# pushes it through the same expanding-window PCA gate as MMR, and feeds the
# orthogonal factors to the walk-forward. NOTE: the macro panel only has ~5y of
# real history (yfinance), so for older dates the macro columns are flat-filled
# and the factors are effectively driven by the sheet predictors alone.
# Empirically OFF: feeding macros into Aarambh degraded it in every form tested
# — raw (R²=−4), combined-PCA unstationarized (R²=−47), combined-PCA
# stationarized (R²=−0.78, signal INVERTED). Macros add no fair-value signal for
# PE and actively harm the model. Kept as a toggle for A/B, default off.
# (Also INERT in predictive mode — the combined-PCA gate is bypassed there — so
# False is the safe default in both modes: neutral when forward-on, correct when
# forward-off. Verified: app PCA=True fwd IC +0.131 ≈ standalone no-PCA +0.128.)
AARAMBH_PCA_PREDICTORS = False
AARAMBH_PCA_N_COMPONENTS = 9

# When True, the 44 custom engineered predictors (yield spreads, real rates,
# credit/commodity ratios, FX momentum, cross-asset composites — see
# docs/CUSTOM_PREDICTORS.md / analytics/custom_features.py) are computed and fed
# into both engines via the causal PCA gate (all 44 → MMR; 41 → Aarambh, with
# the 3 PE/PB/DY-embedding ones excluded to avoid target leakage). Toggle off
# to A/B against the base feature set. All features are causal & non-repainting.
CUSTOM_PREDICTORS_ENABLED = True

# Predictors RESERVED as passthrough — kept as their own raw, named columns in
# the feed and NOT compressed into the PCA factors, so their direct India
# rate/inflation signal isn't diluted across the orthogonal components. Applied
# to both the Aarambh combined PCA and the MMR macro-factor PCA. Still causal
# (raw data is point-in-time).
#   PCA_PASSTHROUGH_ENABLED — master on/off toggle (like AARAMBH_PCA_PREDICTORS).
#   PCA_PASSTHROUGH         — which columns to reserve when enabled.
PCA_PASSTHROUGH_ENABLED = True
PCA_PASSTHROUGH = ("IN10Y", "IN02Y", "IN30Y")

# What forward series the Intelligence layer (ConvergenceTuner calibration AND
# the directional convergence test) is optimized to predict. Drives BOTH so the
# whole pipeline stays coherent with the selected engine target.
#   "target" — the SELECTED target's forward change (e.g. NIFTY50_PE). Coherent
#              with what Aarambh models; over the 3–20d calibration horizons
#              earnings are ~flat, so ΔlogPE ≈ price return anyway.
#   "nsei"   — NIFTY 50 index (^NSEI) forward PRICE return: the true,
#              survivorship-correct benchmark. Use when you want a tradeable
#              price signal regardless of the engine target.
#   "basket" — equal-weighted constituent basket return (a real return, but
#              survivorship-biased — built from TODAY's members).
# "nsei"/"basket" fall back to each other, then to "target", if data is missing.
#
# DATA-BACKED DEFAULT = "nsei" (2026-06-15 study, real ConvergenceTuner, 50
# trials each, predictive mode). Calibrating the convergence weights against the
# forward label gave:
#     label      val IC   fold-stab   walk-forward (re-cal, purged OOS)
#     target=PE  +0.148   100%        100% durable  (+0.156)
#     nsei       +0.133   100%        100% durable  (+0.137)
#     basket     +0.038    80%         67% NOT durable
# The survivorship-biased BASKET poisons the calibration (train IC 0.163 ≫ val
# 0.038 — it overfits survivor drift), which is what produced every prior run's
# "walk-forward NOT durable" verdict. ^NSEI and PE both give a barely-fit,
# strongly-generalizing, fully-durable calibration. ^NSEI is chosen as default:
# it is the survivorship-correct PRICE return — the asset actually traded — and
# is ~as strong as PE (which is marginally higher but a valuation proxy, not
# tradeable). NEVER default to "basket". In predictive mode "target" now uses the
# real target LEVEL (not the basket fallback) — see app.py label resolution.
CALIBRATION_RETURN_LABEL = "nsei"

# ── Experimental: predictive returns mode for Aarambh (default OFF) ──────────
# When True, Aarambh stops regressing the PE *level* (a near-tautological fit
# with no forward edge — see docs/FINDINGS.md) and instead FORECASTS the forward
# AARAMBH_FWD_HORIZON-day change of the target from trailing AARAMBH_FWD_MOM_K-day
# momentum of the predictors (an ex-ante setup). Conviction becomes the forecast
# itself (−prediction → bullish pole), and R²-vs-RW / IC then measure genuine
# out-of-sample forecast skill rather than persistence. This is a research head:
# every levels sweep shows no edge, so this tests whether a returns forecast
# carries IC the levels model structurally cannot. Bypasses the combined-PCA
# predictor gate (uses raw momentum features). Off by default — it changes the
# engine's thesis from mean-reversion-to-fair-value to momentum forecasting.
AARAMBH_FORWARD_SIGNAL = True
AARAMBH_FWD_HORIZON = 10   # forecast horizon (trading days)
AARAMBH_FWD_MOM_K = 20     # trailing momentum window for the predictor features

# Signal thresholds (conviction score → signal mapping)
CONVICTION_STRONG = 60
CONVICTION_MODERATE = 40
CONVICTION_WEAK = 20

# Z-score zone boundaries
Z_EXTREME = 2.0
Z_THRESHOLD = 1.0

# Staleness
STALENESS_DAYS = 3

# Timeframe filter mapping (trading days)
TIMEFRAME_TRADING_DAYS = {"3M": 63, "6M": 126, "1Y": 252, "2Y": 504}

# Default predictors for NIFTY50 use case
DEFAULT_PREDICTORS = (
    "AD_RATIO", "COUNT", "REL_AD_RATIO", "REL_BREADTH",
    "IN10Y", "IN02Y", "IN30Y", "INIRYY", "REPO",
    "US02Y", "US10Y", "US30Y", "NIFTY50_DY", "NIFTY50_PB",
)

# Google Sheets URL (should be set via secrets or environment variable)
# This is only a placeholder for type hints
DEFAULT_SHEET_URL = ""

# DDM parameters (calibrated for daily conviction series)
DDM_LEAK_RATE = 0.08
DDM_DRIFT_SCALE = 0.15
DDM_LONG_RUN_VAR = 100.0

# ─── Nirnay Engine Defaults ──────────────────────────────────────────────────

NIRNAY_MSF_LENGTH = 20
NIRNAY_ROC_LEN = 14
NIRNAY_REGIME_SENSITIVITY = 1.0
NIRNAY_BASE_WEIGHT = 0.6
NIRNAY_MMR_NUM_VARS = 5

# Nirnay signal thresholds (oscillator scale: -10 to +10)
NIRNAY_OVERSOLD = -5
NIRNAY_OVERBOUGHT = 5
NIRNAY_STRONG_BUY = -7
NIRNAY_STRONG_SELL = 7

# ─── Convergence Layer Defaults ──────────────────────────────────────────────

# Adaptive weighting base allocation
CONV_WEIGHT_DIRECTION = 0.30
CONV_WEIGHT_BREADTH = 0.25
CONV_WEIGHT_MAGNITUDE = 0.25
CONV_WEIGHT_REGIME = 0.20

# Adaptive shift limits (±10% based on clarity ratios)
CONV_ADAPTIVE_SHIFT_MAX = 0.10

# Convergence zone thresholds
CONV_STRONG_BULLISH = -60
CONV_MODERATE_BULLISH = -30
CONV_WEAK_BULLISH = -10
CONV_WEAK_BEARISH = 10
CONV_MODERATE_BEARISH = 30
CONV_STRONG_BEARISH = 60

# DDM for convergence score
CONV_DDM_LEAK_RATE = 0.10
CONV_DDM_DRIFT_SCALE = 0.12
CONV_DDM_LONG_RUN_VAR = 50.0

# Divergence detection
DIV_LOOKBACK = 20
DIV_PERSISTENCE_THRESHOLD = 5

# ─── Column Normalization ────────────────────────────────────────────────────

# ─── Global Macro Bond ETF Universe ──────────────────────────────────────────
# Adapted from Sanket — proxy for global yield dynamics via yfinance-available
# bond ETFs. Replaces the (now-broken) Stooq direct yield endpoints.
# Yields the same macro signal Stooq did, but via a stable yfinance source.

GLOBAL_MACRO_MAP = {
    # ── US Treasuries (Full Curve) ─────────────────────────────────────────
    "US Treasury 1-3 Month":             "BIL",
    "US Treasury Ultra-Short (0-1Y)":    "SHV",
    "US Treasury 0-3 Month (SGOV)":      "SGOV",
    "US Treasury Short (1-3Y)":          "SHY",
    "US Treasury Short (1-3Y) Vanguard": "VGSH",
    "US Treasury Intermediate (3-7Y)":   "IEI",
    "US Treasury Intermediate (7-10Y)":  "IEF",
    "US Treasury Intermediate Vanguard": "VGIT",
    "US Treasury Long (10-20Y)":         "TLH",
    "US Treasury Long (20Y+)":           "TLT",
    "US Treasury Long Vanguard":         "VGLT",
    "US Treasury Total Market":          "GOVT",
    # ── Direct Yield Indices (Raw %) ───────────────────────────────────────
    "US 13-Week T-Bill Yield":           "^IRX",
    "US 5-Year Treasury Yield":          "^FVX",
    "US 10-Year Treasury Yield":         "^TNX",
    "US 30-Year Treasury Yield":         "^TYX",
    # ── Inflation-Protected (TIPS) ─────────────────────────────────────────
    "US TIPS Broad Market":              "TIP",
    "US TIPS Short-Term":                "VTIP",
    "International Govt Inflation-Linked": "WIP",
    # ── Aggregate / Multi-Sector ───────────────────────────────────────────
    "US Core Aggregate Bond":            "AGG",
    "US Total Bond Market":              "BND",
    "US Floating Rate Notes":            "FLOT",
    "Global Aggregate Bond (Hedged)":    "BNDW",
    "Total International Bond (ex-US)":  "BNDX",
    # ── US Corporate: Investment Grade ─────────────────────────────────────
    "US Corporate Investment Grade":     "LQD",
    "US Corporate Short-Term (1-5Y)":    "VCSH",
    "US Corporate Intermediate":         "VCIT",
    "US Corporate Long-Term":            "VCLT",
    # ── High Yield & Alternative Credit ────────────────────────────────────
    "US High Yield Corporate":           "HYG",
    "US High Yield Corporate SPDR":      "JNK",
    "Global High Yield Bond":            "GHYG",
    "Global Green Bond":                 "BGRN",
    "Preferred Stock (Hybrid)":          "PFF",
    "Convertible Bonds":                 "CWB",
    "Fallen Angels (Recent HY)":         "FALN",
    # ── Structured & Asset-Backed ──────────────────────────────────────────
    "US Mortgage-Backed Securities":     "MBB",
    "US Mortgage-Backed Vanguard":       "VMBS",
    "US Senior Loan (Floating Rate)":    "BKLN",
    # ── Municipal Bonds ────────────────────────────────────────────────────
    "US Municipal National":             "MUB",
    "US Municipal Tax-Exempt Vanguard":  "VTEB",
    # ── Developed Markets Sovereign (Europe) ───────────────────────────────
    "International Treasury (ex-US)":    "IGOV",
    "International Treasury SPDR":       "BWX",
    "International Corporate Bonds":     "IBND",
    "Eurozone Government Bond":          "IEGA.L",
    "Eurozone Corporate Bond (IG)":      "IEAC.L",
    "Germany Short-Term (Schatz)":       "SDEU.L",
    "UK Gilts":                          "IGLT.L",
    "UK Gilts (Inflation-Linked)":       "INXG.L",
    "UK Corporate Bonds":                "SLXX.L",
    # ── Developed Markets Sovereign (Asia-Pacific) ─────────────────────────
    "Japan Government Bonds (Broad)":    "JGBL.L",
    "Australia Government Bonds":        "VGB.AX",
    "Canada Broad Aggregate Bond":       "XBB.TO",
    # ── India Fixed Income ─────────────────────────────────────────────────
    "India Gov Bonds (LSE Proxy)":       "IIND.L",
    "India 8-13Y G-Sec":                 "LTGILTBEES.NS",
    "India 5Y G-Sec":                    "GILT5YBEES.NS",
    "India AAA PSU Bond (Bharat 2030)":  "EBBETF0430.NS",
    "India Overnight Rate (Liquid)":     "LIQUIDBEES.NS",
    # ── Emerging Markets ───────────────────────────────────────────────────
    "EM Sovereign Debt (USD)":           "EMB",
    "EM Sovereign Debt USD Invesco":     "PCY",
    "EM Sovereign (Local Currency)":     "EMLC",
    "EM High Yield Corporate":           "EMHY",
    "China Government Bonds":            "CBON",
    "China CNY Local Bonds":             "CNYB.L",
    # ── Broad Duration Proxies ─────────────────────────────────────────────
    "Short-Term Broad Bond":             "BSV",
    "Long-Term Broad Bond":              "BLV",
}

# Yahoo Finance macro symbols — commodities and FX, fetched alongside Global Macro.
MACRO_SYMBOLS_YF = {
    # Major FX
    "Dollar Index": "DX-Y.NYB",
    "USD/INR": "INR=X",
    "EUR/INR": "EURINR=X",
    "GBP/INR": "GBPINR=X",
    "JPY/INR": "JPYINR=X",
    # Commodities - Metals
    "Gold": "GC=F",
    "Silver": "SI=F",
    "Copper": "HG=F",
    "Platinum": "PL=F",
    # Commodities - Energy
    "Crude Oil": "CL=F",
    "Brent Crude": "BZ=F",
    "Natural Gas": "NG=F",
    # Commodities - Agriculture
    "Corn": "ZC=F",
    "Wheat": "ZW=F",
    "Soybeans": "ZS=F",
    "Cotton": "CT=F",
    "Coffee": "KC=F",
    "Sugar": "SB=F",
}

# ─── Nifty 50 Constituents ──────────────────────────────────────────────────

NIFTY50_URL = "https://www.niftyindices.com/IndexConstituent/ind_nifty50list.csv"
NIFTY_INDICES = {
    "NIFTY 50": "https://www.niftyindices.com/IndexConstituent/ind_nifty50list.csv",
    "NIFTY NEXT 50": "https://www.niftyindices.com/IndexConstituent/ind_niftynext50list.csv",
    "NIFTY 100": "https://www.niftyindices.com/IndexConstituent/ind_nifty100list.csv",
    "NIFTY 200": "https://www.niftyindices.com/IndexConstituent/ind_nifty200list.csv",
    "NIFTY 500": "https://www.niftyindices.com/IndexConstituent/ind_nifty500list.csv",
}

# ─── Chart Theme ─────────────────────────────────────────────────────────────

CHART_BG = "rgba(0,0,0,0)"
CHART_GRID = "rgba(255,255,255,0.03)"
CHART_ZEROLINE = "rgba(255,255,255,0.08)"
CHART_FONT_COLOR = "#94A3B8"

# Signal colors - Obsidian Quant
COLOR_GREEN = "#34D399"  # EMERALD
COLOR_RED = "#FB7185"    # ROSE
COLOR_GOLD = "#D4A853"   # AMBER GOLD
COLOR_CYAN = "#22D3EE"   # CYAN
COLOR_AMBER = "#D4A853"  # AMBER
COLOR_PURPLE = "#A78BFA"  # VIOLET (matches CSS --violet)
COLOR_MUTED = "rgba(148,163,184,0.4)"  # SLATE

# ─── UI Thresholds (centralized magic numbers) ──────────────────────────────

# Conviction score thresholds for signal classification
UI_CONVICTION_STRONG = 60
UI_CONVICTION_MODERATE = 40
UI_CONVICTION_WEAK = 20

# Z-score thresholds for extreme values
UI_Z_EXTREME = 2.0
UI_Z_THRESHOLD = 1.0

# Breadth percentage thresholds
UI_BREADTH_HIGH = 60  # % threshold for high breadth alert

# Agreement ratio thresholds
UI_AGREEMENT_STRONG = 0.7
UI_AGREEMENT_MODERATE = 0.5

# Nirnay avg signal thresholds
UI_NIRNAY_BULLISH = -2
UI_NIRNAY_BEARISH = 2

# Model spread thresholds
UI_MODEL_SPREAD_LOW = 0.5
UI_MODEL_SPREAD_HIGH = 1.5

# OOS R² thresholds
UI_R2_STRONG = 0.7
UI_R2_ACCEPTABLE = 0.4

# Band width interpretation
UI_BAND_NARROW = 30
UI_BAND_WIDE = 60

# HMM probability threshold
UI_HMM_CONFIDENT = 0.5

# ADF/KPSS p-value thresholds
UI_ADF_SIGNIFICANT = 0.05
UI_KPSS_NOT_SIGNIFICANT = 0.05

# Chart height defaults
UI_CHART_HEIGHT_SMALL = 280
UI_CHART_HEIGHT_MEDIUM = 340
UI_CHART_HEIGHT_LARGE = 380
UI_CHART_HEIGHT_XLARGE = 540
UI_CHART_HEIGHT_STACKED = 680

# Data table defaults
UI_TABLE_HEIGHT = 520
UI_TABLE_HISTORY_ROWS = 10

# ─── Nifty 50 Default Constituents (fallback list) ──────────────────────────

NIFTY50_FALLBACK = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
    "HINDUNILVR.NS", "SBIN.NS", "BHARTIARTL.NS", "BAJFINANCE.NS", "KOTAKBANK.NS",
    "LTM.NS", "HCLTECH.NS", "ITC.NS", "AXISBANK.NS", "ASIANPAINT.NS",
    "MARUTI.NS", "SUNPHARMA.NS", "TITAN.NS", "ULTRACEMCO.NS", "WIPRO.NS",
    "ADANIENT.NS", "ADANIPORTS.NS", "APOLLOHOSP.NS", "BRITANNIA.NS", "CIPLA.NS",
    "COALINDIA.NS", "DIVISLAB.NS", "DRREDDY.NS", "EicherMot.NS", "GRASIM.NS",
    "HDFCLIFE.NS", "HEROMOTOCO.NS", "JSWSTEEL.NS", "LTIM.NS", "M&M.NS",
    "NESTLEIND.NS", "ONGC.NS", "POWERGRID.NS", "SBILIFE.NS", "SHRIRAMFIN.NS",
    "TECHM.NS", "TATACONSUM.NS", "TATAMOTORS.NS", "TATASTEEL.NS", "UPL.NS",
    "BAJAJFINSV.NS", "BPCL.NS", "INDUSINDBK.NS", "NAUKRI.NS", "PIDILITIND.NS",
]

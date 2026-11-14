"""
Nishkarsh — Configuration: the single source of truth for every constant, threshold
and default. निष्कर्ष (Nishkarsha) — "Conclusion / Inference".

Rules:
  * No other module may hard-code a value that lives here.
  * Test-mode values are injected through function arguments / fixtures, NEVER by
    editing this file (the v1.6.0 upload shipped test-mode 15/30 training windows
    from this file — see CHANGELOG 2.0.0 Erratum).
  * ``validate_config()`` (bottom of file) runs at import and fails loudly on an
    incoherent configuration.
"""

# ─── Version / Product ───────────────────────────────────────────────────────

VERSION = "2.5.10"
PRODUCT_NAME = "NISHKARSH"
COMPANY = "@thebullishvalue"

# ─── Aarambh Engine Defaults ─────────────────────────────────────────────────

LOOKBACK_WINDOWS = (5, 10, 20, 50, 100)
# Walk-forward training window, in LABEL-COMPLETE trading-day rows (rows whose
# forward label is fully realized before the forecast point — the purge gap is
# applied on top, never instead). The engine fits a ~14-feature regularized
# ensemble, so the window must be >> feature count to stay well-conditioned.
# MIN bootstraps the expanding window at ~6 trading years; MAX caps it at ~8 so
# the model stays adaptive to regime drift. (History: 15/30 originally —
# degenerate p/n whose apparent edge was an overlapping-label artifact; 252/756
# after the 1.4.22 sweep; 1500/2000 from 1.4.22 on. v1.6.0 accidentally shipped
# 15/30, under which the forward-label purge never applied — restored in 2.0.0.)
MIN_TRAIN_SIZE = 1500
MAX_TRAIN_SIZE = 2000
# Fixed walk-forward refit cadence (rows between ensemble refits). A constant —
# a cadence derived from n made every forecast repaint whenever n crossed a
# threshold (audit A1-14).
AARAMBH_WF_REFIT = 5
# Exponential sample-weight half-life (trading days) inside each training window.
AARAMBH_SAMPLE_HALFLIFE = 252
# Purged inner CV for the Ridge / ElasticNet regularization search
# (TimeSeriesSplit with gap = forecast horizon; LOO on overlapping labels
# under-regularizes — audit A1-15).
AARAMBH_INNER_CV_SPLITS = 3
# Widened upward: on 1500–2000-row windows the old ceiling (100) was binding.
RIDGE_ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0)
HUBER_EPSILON = 1.35
# 200 is bit-identical to 500 here: on real data the walk-forward Huber fits
# converge well before the cap (0 of 140 chunks hit 150 iters), and OOS R²/IC
# are unchanged from max_iter 200→500. Trimmed purely to drop wasted headroom.
HUBER_MAX_ITER = 200
OU_PROJECTION_DAYS = 90
# Predictor history guard: a predictor still NaN anywhere in the most recent
# MIN_DATA_POINTS rows cannot support the walk-forward and is dropped.
MIN_DATA_POINTS = 1500
# Minimum number of genuine out-of-sample forecast rows a run must produce.
# Total rows required = MIN_TRAIN_SIZE + AARAMBH_FWD_HORIZON (purge) + MIN_OOS_ROWS.
MIN_OOS_ROWS = 252

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

# ── Predictive returns mode for Aarambh (SHIPPED DEFAULT: ON) ────────────────
# When True, Aarambh stops regressing the PE *level* (a near-tautological fit
# with no forward edge — see docs/FINDINGS.md) and instead FORECASTS the forward
# AARAMBH_FWD_HORIZON-day change of the target from trailing AARAMBH_FWD_MOM_K-day
# momentum of the predictors (an ex-ante setup). Conviction becomes the forecast
# itself (−prediction → bullish pole), and R²-vs-RW / IC then measure genuine
# out-of-sample forecast skill rather than persistence. This is a research head:
# every levels sweep shows no edge, so this tests whether a returns forecast
# carries IC the levels model structurally cannot. Bypasses the combined-PCA
# predictor gate (uses raw momentum features). SHIPPED DEFAULT: ON — the
# levels model demonstrated no edge in every sweep (docs/FINDINGS.md), so the
# returns-forecast head is the live thesis; set False to A/B the legacy
# fair-value levels regression.
AARAMBH_FORWARD_SIGNAL = True
AARAMBH_FWD_HORIZON = 10   # forecast horizon (trading days)
AARAMBH_FWD_MOM_K = 20     # trailing momentum window for the predictor features

# ── Precedent (historical analog) view ───────────────────────────────────────
# The Precedent tab matches today's state to the most statistically-similar past
# states (covariance-aware Mahalanobis on engine.ts_data) and reports what the
# target did next — a descriptive base rate complementing the Aarambh forecast.
# Forward-return HOLD grid for the base-rate summary; the primary read is the
# Aarambh forecast horizon (AARAMBH_FWD_HORIZON), momentum window AARAMBH_FWD_MOM_K.
PRECEDENT_HOLD_HORIZONS = (3, 5, 10, 20)

# Signal thresholds (conviction score → signal mapping)
CONVICTION_STRONG = 60
CONVICTION_MODERATE = 40
CONVICTION_WEAK = 20

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

# DDM parameters (calibrated for daily conviction series)
DDM_LEAK_RATE = 0.08
DDM_DRIFT_SCALE = 0.15
DDM_LONG_RUN_VAR = 100.0

# ─── Nirnay Engine Defaults ──────────────────────────────────────────────────

NIRNAY_MSF_LENGTH = 20
NIRNAY_ROC_LEN = 14
# 1.5 preserves the value the engine ACTUALLY ran with — app.py had these
# knobs hardcoded (audit D2) with 1.5 while this constant said 1.0; the
# constants are now genuinely the single source of truth (wired through the
# engine call and the run log), set to match shipped behavior.
NIRNAY_REGIME_SENSITIVITY = 1.5
NIRNAY_BASE_WEIGHT = 0.6
NIRNAY_MMR_NUM_VARS = 5

# Online-HMM emission std floor. 1e-4 let a state's emission collapse onto a
# frozen mean (50/50 synthetic stocks), making regime labels path/start-date
# dependent; 0.10 removes the collapse (audit A2-01).
NIRNAY_HMM_STD_FLOOR = 0.10
# First N genuine (non-NaN Close) bars of each constituent are warm-up: flagged
# `Warmup`, excluded from daily breadth and from the convergence layer (A2-03/13).
NIRNAY_WARMUP_BARS = 50

# ─── Nirnay polarity — the ONE switch for how bottom-up readings map to direction ─
# "trend":      strength reads bullish — a high constituent oscillator, overbought breadth,
#               bull-regime / bull-HMM majorities and overbought-side cut crossings count as
#               + (bullish), the same "market is strong" orientation as Aarambh's forecast.
# "contrarian": the reverse (oversold / bear-regime = bullish) — Nirnay's original
#               mean-reversion design, used through 2.2.0.
# Every Nirnay-derived quantity reads this: the Nirnay head zN, agreement anatomy,
# consensus / agreement state, Convergence-Lift and κ, divergence typing, the signed
# agreement oscillator and the Nishkarsh Oscillator's bottom-up pillar.
# OWNER DECISION (2026-11-09): "trend". Recorded evidence measured under "contrarian"
# (docs/research/*_RESULTS.md) showed Aarambh's accuracy +0.08 when contrarian-Nirnay
# confirmed vs −0.26 when it contradicted; under "trend" those two buckets swap.
# Re-measured under "trend" in docs/research/NIRNAY_POLARITY_CHANGE.md.
NIRNAY_POLARITY = "trend"
NIRNAY_SIGN = 1.0 if NIRNAY_POLARITY == "trend" else -1.0

# Nirnay signal thresholds (oscillator scale: -10 to +10).
# NOTE: these are COLD-START PRIORS / flat-band fallbacks only. The live
# oversold/overbought cuts are each stock's causal expanding 30th/70th
# percentile of its own Unified oscillator (≈ ±1.9 in practice), so ~30% / 30%
# of stock-days are flagged oversold / overbought by construction.
NIRNAY_OVERSOLD = -5
NIRNAY_OVERBOUGHT = 5

# ─── Convergence Layer Defaults ──────────────────────────────────────────────

# Agreement-anatomy base weights (descriptive only — convergence/cross_validator.py;
# NOT used in the conclusion, no adaptive shift, no calibration).
CONV_WEIGHT_DIRECTION = 0.30
CONV_WEIGHT_BREADTH = 0.25
CONV_WEIGHT_MAGNITUDE = 0.25
CONV_WEIGHT_REGIME = 0.20

# ─── Conclusion Layer v2 — "Two systems. One conclusion." as a MEASURED claim ─
# Aarambh's forecast is the directional head (causally z-scored). Nirnay is a
# conditioning variable: it can only SHRINK conviction (×κ ≤ 1) on days it
# contradicts Aarambh, and only after a Convergence-Lift test shows that
# Aarambh is measurably more accurate when the two systems agree than when they
# contradict (out-of-sample, non-overlapping, one-sided Fisher-z test).
# See docs/ARCHITECTURE.md and convergence/conclusion.py.
CONV2_TAU = 0.25                 # |zN| needed for Nirnay to CONFIRM / CONTRADICT (fixed, not fitted)
CONV2_ZMIN_PERIODS = 126         # min prior observations before a causal z-score exists
CONV2_ZCLIP = 4.0                # causal z-score clip
CONV2_REFIT_EVERY = 126          # walk-forward κ refit cadence (rows)
CONV2_MIN_HISTORY = 500          # rows of overlap before the first κ refit
CONV2_LIFT_ALPHA = 0.05          # one-sided significance for the Convergence-Lift test
CONV2_MIN_BUCKET = 30            # min h-spaced windows in each of CONFIRM / CONTRADICT
CONV2_DIV_EXTREME = 1.5          # |z| for an engine to be "leading" a divergence
CONV2_DIV_CALM = 0.5             # |z| below which the other engine counts as calm
CONV2_DIV_PERSIST = (5, 20)      # CONTRADICTION: ≥5 CONTRADICT days in the last 20 trading days
# Fixed conclusion tiers on C = 100·tanh(zA/2)·m  (positive = bullish).
CONV2_TIER_STRONG = 60.0         # |zA| ≈ 1.39
CONV2_TIER_MODERATE = 30.0       # |zA| ≈ 0.62
CONV2_TIER_WEAK = 10.0

# ─── Edge evidence (one scale for UI, run log and hero) ─────────────────────
EDGE_ALPHA = 0.05                # significance for "edge" claims (Fisher-z / HAC)
EDGE_TRADEABLE_T = 2.0           # net-of-cost HAC t needed for "TRADEABLE EDGE"
EDGE_BOOTSTRAP_REPS = 2000       # stationary block bootstrap replications (seeded)
EDGE_BOOTSTRAP_SEED = 20260704
BACKTEST_COST_BPS = (0.0, 2.0, 5.0, 10.0)  # one-way cost grid (bps per unit turnover)
BACKTEST_REFERENCE_BPS = 5.0     # the cost level verdicts are graded at

# Sheet columns that hold the NIFTY 50 index LEVEL (IST-dated). When the live ^NSEI
# fetch is unavailable, the first one present is the return label — preferred over the
# PE proxy because it is the traded price, and it spans the sheet's full history.
INDEX_SHEET_COLUMNS = ("NIFTY", "NIFTY50", "NIFTY_50", "NSEI")

# ─── Pipeline / data window ──────────────────────────────────────────────────
# yfinance history (constituents + macro). The start is ANCHORED to 1 January of
# (current year − HISTORY_YEARS) so the window slides once a year, not daily — a
# daily-sliding start repainted expanding statistics / PCA / HMM history on every
# run (audit A2-06). NOTE: constituents are TODAY's index members, so a longer
# window adds survivorship bias; 7y is the deliberate ceiling.
HISTORY_YEARS = 7
# Shift non-IST-synchronous macro columns (US / EU / FX / commodities, whose
# close for date D prints after the NSE close of D) by one NSE session before
# joining them onto NSE dates — removes a same-date look-ahead (audit A2-04).
MACRO_LAG_NON_IST = True
# Tickers / columns that ARE synchronous with the NSE close (never lagged).
IST_SYNCHRONOUS_SUFFIXES = (".NS", ".BO")
IST_SYNCHRONOUS_COLUMNS = ("IN10Y", "IN02Y", "IN30Y", "INIRYY", "REPO",
                           "^NSEI", "^NSEBANK", "^INDIAVIX",
                           "AD_RATIO", "COUNT", "REL_AD_RATIO", "REL_BREADTH",
                           "NIFTY50_PE", "NIFTY50_PB", "NIFTY50_DY")
# Google-Sheet predictor columns quoting a NON-Indian market close for date D
# (prints after the NSE close of D). Lagged one row before they enter EITHER
# engine. Judgement call: if the sheet already records these with a one-day
# lag, set this to () — the sheet's provenance is not visible to the code.
NON_IST_SHEET_COLUMNS = ("US02Y", "US10Y", "US30Y")
# Duplicated rate series: when the sheet carries the left column, the yfinance ticker on
# the right measures the same yield and is dropped from the macro panel (no double count).
MACRO_DUPLICATE_RATES = {"US10Y": "^TNX", "US30Y": "^TYX"}
# Causal macro-factor PCA for Nirnay's MMR: applied when the macro panel has more than
# MACRO_FACTOR_MIN_COLUMNS columns; MACRO_FACTOR_COMPONENTS orthogonal factors (+ passthrough).
MACRO_FACTOR_COMPONENTS = 8
MACRO_FACTOR_MIN_COLUMNS = 12
# Sheet columns fed (with the macro panel) to the custom-feature builder only.
CUSTOM_FEATURE_SHEET_COLUMNS = ("REPO", "INIRYY", "AD_RATIO", "REL_AD_RATIO", "REL_BREADTH",
                                "NIFTY50_PE", "NIFTY50_DY", "NIFTY50_PB",
                                "IN10Y", "IN02Y", "IN30Y", "US10Y", "US02Y", "US30Y")
# Sheet bond-yield columns joined into Nirnay's macro panel.
MACRO_SHEET_BOND_COLUMNS = ("IN10Y", "IN02Y", "IN30Y", "US10Y", "US30Y", "US02Y")

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

# ─── Chart Theme ─────────────────────────────────────────────────────────────

# Signal colors - Obsidian Quant
COLOR_GREEN = "#34D399"  # EMERALD
COLOR_RED = "#FB7185"    # ROSE
COLOR_CYAN = "#22D3EE"   # CYAN
COLOR_AMBER = "#D4A853"  # AMBER
COLOR_PURPLE = "#A78BFA"  # VIOLET (matches CSS --violet)
COLOR_MUTED = "rgba(148,163,184,0.4)"  # SLATE

# ─── UI layout ───────────────────────────────────────────────────────────────

# Chart height defaults
UI_CHART_HEIGHT_SMALL = 280
UI_CHART_HEIGHT_MEDIUM = 340
UI_CHART_HEIGHT_LARGE = 380
UI_CHART_HEIGHT_XLARGE = 540
UI_CHART_HEIGHT_STACKED = 680

# ─── Nifty 50 Default Constituents (fallback list) ──────────────────────────
# Used ONLY when the live niftyindices.com list (and its cached copy) is
# unavailable; every run that uses it says so (constituent provenance
# "fallback"). Membership reflects the index as of NIFTY50_FALLBACK_AS_OF
# (incl. the 2024–25 changes: TRENT/BEL, ETERNAL/JIOFIN, MAXHEALTH/INDIGO,
# and TMPV after the Tata Motors demerger). Index membership changes twice a
# year — verify against niftyindices.com and update both constants together.
NIFTY50_FALLBACK_AS_OF = "2025-10-01"
NIFTY50_FALLBACK = [
    "ADANIENT.NS", "ADANIPORTS.NS", "APOLLOHOSP.NS", "ASIANPAINT.NS", "AXISBANK.NS",
    "BAJAJ-AUTO.NS", "BAJFINANCE.NS", "BAJAJFINSV.NS", "BEL.NS", "BHARTIARTL.NS",
    "CIPLA.NS", "COALINDIA.NS", "DRREDDY.NS", "EICHERMOT.NS", "ETERNAL.NS",
    "GRASIM.NS", "HCLTECH.NS", "HDFCBANK.NS", "HDFCLIFE.NS", "HINDALCO.NS",
    "HINDUNILVR.NS", "ICICIBANK.NS", "INDIGO.NS", "INFY.NS", "ITC.NS",
    "JIOFIN.NS", "JSWSTEEL.NS", "KOTAKBANK.NS", "LT.NS", "M&M.NS",
    "MARUTI.NS", "MAXHEALTH.NS", "NESTLEIND.NS", "NTPC.NS", "ONGC.NS",
    "POWERGRID.NS", "RELIANCE.NS", "SBILIFE.NS", "SBIN.NS", "SHRIRAMFIN.NS",
    "SUNPHARMA.NS", "TATACONSUM.NS", "TMPV.NS", "TATASTEEL.NS", "TCS.NS",
    "TECHM.NS", "TITAN.NS", "TRENT.NS", "ULTRACEMCO.NS", "WIPRO.NS",
]


# ─── Configuration invariants (fail loudly at import) ───────────────────────

def validate_config() -> None:
    """Raise ``ValueError`` if the configuration is internally incoherent.

    Guards the failure mode that shipped in v1.6.0: test-mode walk-forward
    windows (15/30) edited into this file silently disabled the forward-label
    purge and turned look-ahead into apparent edge.
    """
    errors: list[str] = []
    n_feat = len(DEFAULT_PREDICTORS)
    if MIN_TRAIN_SIZE < 10 * n_feat + AARAMBH_FWD_HORIZON:
        errors.append(
            f"MIN_TRAIN_SIZE={MIN_TRAIN_SIZE} < 10·p + h = {10 * n_feat + AARAMBH_FWD_HORIZON} "
            "(degenerate p/n; overlapping-label fits interpolate)"
        )
    if MAX_TRAIN_SIZE < MIN_TRAIN_SIZE:
        errors.append(f"MAX_TRAIN_SIZE={MAX_TRAIN_SIZE} < MIN_TRAIN_SIZE={MIN_TRAIN_SIZE}")
    if not (1 <= AARAMBH_WF_REFIT <= AARAMBH_FWD_HORIZON):
        errors.append(f"AARAMBH_WF_REFIT={AARAMBH_WF_REFIT} must be in [1, AARAMBH_FWD_HORIZON]")
    if AARAMBH_FWD_HORIZON < 1 or AARAMBH_FWD_MOM_K < 1:
        errors.append("AARAMBH_FWD_HORIZON and AARAMBH_FWD_MOM_K must be ≥ 1")
    if abs(CONV_WEIGHT_DIRECTION + CONV_WEIGHT_BREADTH + CONV_WEIGHT_MAGNITUDE + CONV_WEIGHT_REGIME - 1.0) > 1e-9:
        errors.append("CONV_WEIGHT_* must sum to 1")
    if not (0 < CONV2_TIER_WEAK < CONV2_TIER_MODERATE < CONV2_TIER_STRONG < 100):
        errors.append("CONV2 tiers must satisfy 0 < WEAK < MODERATE < STRONG < 100")
    if not (0 < EDGE_ALPHA < 0.5 and 0 < CONV2_LIFT_ALPHA < 0.5):
        errors.append("EDGE_ALPHA / CONV2_LIFT_ALPHA must be in (0, 0.5)")
    if BACKTEST_REFERENCE_BPS not in BACKTEST_COST_BPS:
        errors.append("BACKTEST_REFERENCE_BPS must be one of BACKTEST_COST_BPS")
    if MIN_OOS_ROWS < 1 or HISTORY_YEARS < 1:
        errors.append("MIN_OOS_ROWS and HISTORY_YEARS must be ≥ 1")
    if MACRO_FACTOR_COMPONENTS < 2 or MACRO_FACTOR_MIN_COLUMNS < MACRO_FACTOR_COMPONENTS:
        errors.append("MACRO_FACTOR_COMPONENTS must be ≥ 2 and ≤ MACRO_FACTOR_MIN_COLUMNS")
    if NIRNAY_POLARITY not in ("trend", "contrarian"):
        errors.append(f"NIRNAY_POLARITY={NIRNAY_POLARITY!r} must be 'trend' or 'contrarian'")
    if CALIBRATION_RETURN_LABEL not in ("nsei", "target", "basket"):
        errors.append(f"CALIBRATION_RETURN_LABEL={CALIBRATION_RETURN_LABEL!r} invalid")
    if errors:
        raise ValueError("Incoherent core/config.py:\n  - " + "\n  - ".join(errors))


validate_config()

"""
Configuration constants, thresholds, column mappings, and shared defaults.
Merged from both Aarambh (correl.py) and Nirnay (nirnay_core.py) monoliths.
"""

# ─── Version / Product ───────────────────────────────────────────────────────

VERSION = "1.1.0"
PRODUCT_NAME = "Nishkarsh"
COMPANY = "@thebullishvalue"

# ─── Aarambh Engine Defaults ─────────────────────────────────────────────────

LOOKBACK_WINDOWS = (5, 10, 20, 50, 100)
MIN_TRAIN_SIZE = 20
MAX_TRAIN_SIZE = 500
REFIT_INTERVAL = 5
RIDGE_ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0)
HUBER_EPSILON = 1.35
HUBER_MAX_ITER = 500
OU_PROJECTION_DAYS = 90
MIN_DATA_POINTS = 80

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
TIMEFRAME_TRADING_DAYS = {"1M": 21, "6M": 126, "1Y": 252, "2Y": 504}

# Default predictors for NIFTY50 use case
DEFAULT_PREDICTORS = (
    "AD_RATIO", "COUNT", "REL_AD_RATIO", "REL_BREADTH",
    "IN10Y", "IN02Y", "IN30Y", "INIRYY", "REPO",
    "US02Y", "US10Y", "US30Y", "NIFTY50_DY", "NIFTY50_PB",
)

# Default Google Sheets URL
DEFAULT_SHEET_URL = (
    "https://docs.google.com/spreadsheets/d/"
    "1po7z42n3dYIQGAvn0D1-a4pmyxpnGPQ13TrNi3DB5_c/"
    "edit?gid=1938234952#gid=1938234952"
)

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

MACRO_COLUMN_MAP = {
    "10YINY.B": "IN10Y",
    "2YINY.B": "IN02Y",
    "30YINY.B": "IN30Y",
    "10YUSY.B": "US10Y",
    "2YUSY.B": "US02Y",
    "30YUSY.B": "US30Y",
    "5YUSY.B": "US05Y",
    "10YUKY.B": "UK10Y",
    "10YDEY.B": "DE10Y",
    "10YCNY.B": "CN10Y",
    "10YJPY.B": "JP10Y",
}

# Nirnay Stooq symbols
MACRO_SYMBOLS_STOOQ = {
    "India 10Y": "10YINY.B",
    "India 02Y": "2YINY.B",
    "US 30Y": "30YUSY.B",
    "US 10Y": "10YUSY.B",
    "US 05Y": "5YUSY.B",
    "US 02Y": "2YUSY.B",
}

# Nirnay Yahoo Finance macro symbols
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

CHART_BG = "#1A1A1A"
CHART_GRID = "#2A2A2A"
CHART_ZEROLINE = "#3A3A3A"
CHART_FONT_COLOR = "#EAEAEA"

# Signal colors
COLOR_GREEN = "#10b981"
COLOR_RED = "#ef4444"
COLOR_GOLD = "#FFC300"
COLOR_CYAN = "#06b6d4"
COLOR_AMBER = "#f59e0b"
COLOR_PURPLE = "#8b5cf6"
COLOR_MUTED = "#888888"

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

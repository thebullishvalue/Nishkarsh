# Custom Predictors — Formulation Catalog (spec, NOT yet implemented)

Engineered, **stationary**, **causal** features for Aarambh (PE fair value) and
Nirnay/MMR (per-stock). Confirmed: `INIRYY` = India CPI inflation YoY.

**Legend**
- **Src:** `S` = sheet (long history, full PE range) · `Y` = yfinance (~5–7y)
- **Aarambh:** ✓ usable · ⚠ recent-only (Y, flat-filled pre-history) · ⊘ circular (embeds PE/PB/DY)
- **Form:** native spread (already stationary) · `z₂₅₂` = causal 252d rolling z-score, clipped ±5 · `Δk` = k-day change/momentum
- **Tier:** ★ core · ◆ strong · • honorary

Macro columns are referenced by **ticker** (matches the fetched panel).

---

## A. Yield curve / term structure  (S · Aarambh ✓)
| Tier | Name | Formula | Thesis | Form |
|---|---|---|---|---|
| ★ | `TERM_SPREAD_IN` | `IN10Y − IN02Y` | India curve slope. Steepening = growth/risk-on → PE expansion; inversion = recession → compression. | native |
| ◆ | `TERM_SPREAD_US` | `US10Y − US02Y` | US slope = global growth/recession regime; drives EM risk appetite. | native |
| ◆ | `CURVE_FULL_IN` | `IN30Y − IN02Y` | Full India curve span (level-independent steepness). | native |
| • | `CURVE_TWIST_IN` | `(IN30Y−IN10Y) − (IN10Y−IN02Y)` | Belly richness / curvature (butterfly). | native |
| • | `LONG_END_IN` | `IN30Y − IN10Y` | Term-premium / fiscal-risk gauge. | native |

## B. Real rates / monetary stance  (S · Aarambh ✓)
| Tier | Name | Formula | Thesis | Form |
|---|---|---|---|---|
| ★ | `REAL_REPO` | `REPO − INIRYY` | Real policy stance. Negative = liquidity-fuelled re-rating (high PE); high positive = valuation headwind. | native |
| ★ | `IN_REAL_10Y` | `IN10Y − INIRYY` | Real long rate — the discount-rate input to equity value. | native |
| ◆ | `POLICY_EXPECT` | `IN02Y − REPO` | Market vs policy short rate. 2Y < repo ⇒ market prices cuts (easing ahead). | native |
| • | `TERM_PREMIUM_IN` | `IN10Y − REPO` | Term premium over policy. | native |

## C. Cross-country differentials / flows  (S · Aarambh ✓)
| Tier | Name | Formula | Thesis | Form |
|---|---|---|---|---|
| ★ | `US_IN_10Y_DIFF` | `US10Y − IN10Y` | Rate differential → FX/portfolio flows. Narrowing = inflows to India; widening = outflow pressure. | native |
| ◆ | `US_IN_2Y_DIFF` | `US02Y − IN02Y` | Short-end carry (RBI vs Fed). | native |
| • | `REL_CURVE_STEEP` | `(IN10Y−IN02Y) − (US10Y−US02Y)` | India vs US growth-expectation differential. | native |

## D. Inflation regime  (S · Aarambh ✓)
| Tier | Name | Formula | Thesis | Form |
|---|---|---|---|---|
| ◆ | `INFLATION_MOM` | `INIRYY` Δ63 | Inflation *acceleration* (direction > level for re-rating). | Δ63 |
| • | `INFLATION_Z` | `z₂₅₂(INIRYY)` | Inflation hot/cold vs own history. | z₂₅₂ |

## E. Breadth / internal momentum  (S · Aarambh ✓)
| Tier | Name | Formula | Thesis | Form |
|---|---|---|---|---|
| ◆ | `BREADTH_MOM` | `REL_BREADTH` Δ10 | Participation thrust — internal market strength. | Δ10 |
| ◆ | `AD_MOM` | `AD_RATIO` Δ21 | Advance/decline momentum. | Δ21 |
| • | `BREADTH_DIVERGENCE` | `REL_AD_RATIO − REL_BREADTH` | Internal breadth divergence (early warning). | native |
| • | `BREADTH_Z` | `z₂₅₂(REL_BREADTH)` | Breadth extreme vs history. | z₂₅₂ |

## F. Equity valuation / risk premium  (S · Aarambh ⊘ circular — Nirnay/convergence only)
| Tier | Name | Formula | Thesis | Form |
|---|---|---|---|---|
| ◆ | `EQ_RISK_PREMIUM` | `(1/NIFTY50_PE) − IN10Y` | Earnings yield − bond yield ("Fed model"). Embeds PE ⇒ **not for Aarambh**; strong for Nirnay/convergence context. | native |
| • | `DY_SPREAD` | `NIFTY50_DY − IN10Y` | Dividend yield vs bond — relative income. | native |
| • | `IMPLIED_ROE` | `NIFTY50_PB / NIFTY50_PE` | = E/B = ROE proxy. Embeds PE ⇒ Nirnay/context only. | native |

## G. Credit risk appetite  (Y · Aarambh ⚠)
| Tier | Name | Formula | Thesis | Form |
|---|---|---|---|---|
| ★ | `CREDIT_HY_IG` | `z₂₅₂(log(HYG/LQD))` | High-yield vs investment-grade — the cleanest global "fear vs greed" gauge. | z₂₅₂ |
| ◆ | `CREDIT_MOM` | `(HYG/LQD)` Δ21 | Direction of risk appetite. | Δ21 |
| ◆ | `EM_CREDIT` | `z₂₅₂(log(EMHY/EMB))` | EM high-yield vs EM sovereign — EM risk premium. | z₂₅₂ |
| • | `FALLEN_ANGELS` | `z₂₅₂(log(FALN/LQD))` | Distressed-credit stress. | z₂₅₂ |
| • | `PREFERRED_STRESS` | `z₂₅₂(log(PFF/LQD))` | Hybrid/financials stress. | z₂₅₂ |

## H. Duration / rate-direction via ETFs  (Y · Aarambh ⚠)
| Tier | Name | Formula | Thesis | Form |
|---|---|---|---|---|
| ◆ | `DURATION_BID` | `z₂₅₂(log(TLT/SHY))` | Flight-to-duration (growth-fear expressed in bonds). | z₂₅₂ |
| ◆ | `US_INFL_EXPECT` | `z₂₅₂(log(TIP/IEF))` | TIPS vs nominal — market inflation expectations / real-yield proxy. | z₂₅₂ |
| • | `GLOBAL_RATES` | `z₂₅₂(log(BNDX/BND))` | ex-US vs US bonds. | z₂₅₂ |

## I. Currency / FX  (Y · Aarambh ⚠)
| Tier | Name | Formula | Thesis | Form |
|---|---|---|---|---|
| ★ | `DXY_MOM` | `DX-Y.NYB` Δ63 | Dollar trend — primary EM-equity headwind. | Δ63 |
| ◆ | `USDINR_MOM` | `INR=X` Δ21 | Rupee depreciation pressure. | Δ21 |
| ◆ | `INR_IDIOSYNCRATIC` | `z₂₅₂(INR=X) − z₂₅₂(DX-Y.NYB)` | Rupee weakness *beyond* broad dollar — India-specific FX stress. | composite z |
| • | `INR_BASKET` | `z₂₅₂(mean(INR=X,EURINR=X,GBPINR=X,JPYINR=X))` | Broad rupee strength/weakness. | z₂₅₂ |

## J. Commodities / growth-inflation  (Y · Aarambh ⚠)
| Tier | Name | Formula | Thesis | Form |
|---|---|---|---|---|
| ★ | `GROWTH_FEAR` | `z₂₅₂(log(GC=F/HG=F))` | Gold/Copper — growth-vs-fear barometer. High = defensive (PE compression). | z₂₅₂ |
| ◆ | `OIL_SHOCK` | `CL=F` Δ21 | Crude momentum — India oil-import shock (inflation/CAD/margins). | Δ21 |
| ◆ | `COPPER_MOM` | `HG=F` Δ21 | Dr. Copper — global cyclical demand. | Δ21 |
| • | `GOLD_INR` | `z₂₅₂(log(GC=F × INR=X))` | Domestic safe-haven (gold + currency). | z₂₅₂ |
| • | `AGRI_INFLATION` | `z₂₅₂(mean(ZW=F,ZC=F,ZS=F,SB=F))` | Food/agri inflation (India CPI-sensitive). | z₂₅₂ |
| • | `PRECIOUS_RATIO` | `z₂₅₂(log(GC=F/SI=F))` | Gold/Silver — deflation/risk gauge. | z₂₅₂ |
| • | `ENERGY_METALS` | `z₂₅₂(log(CL=F/HG=F))` | Cost-push (energy) vs demand-pull (metals). | z₂₅₂ |

## K. EM / China  (Y · Aarambh ⚠)
| Tier | Name | Formula | Thesis | Form |
|---|---|---|---|---|
| • | `EM_FX_CARRY` | `z₂₅₂(log(EMLC/EMB))` | EM local-ccy vs USD sovereign — EM currency risk. | z₂₅₂ |
| • | `CHINA_RATES` | `(CNYB.L)` Δ21 | China bond regime — regional spillover. | Δ21 |

## L. Cross-asset composites  (mixed)
| Tier | Name | Formula | Thesis | Form |
|---|---|---|---|---|
| ◆ | `RISK_ON_OFF` | `mean(−CREDIT_HY_IG, −GROWTH_FEAR, −DXY_MOM_z, −DURATION_BID)` | Single cross-asset risk-appetite index (↑ = risk-on). | composite z |
| ◆ | `FIN_CONDITIONS` | `mean(−REAL_REPO_z, −CREDIT_HY_IG, −DXY_z, +TERM_SPREAD_IN_z)` | Financial-conditions index (FCI proxy; ↑ = easier). | composite z |

---

## Recommended curated set (avoid feature explosion)

Feeding all ~45 would wreck p/n. A disciplined set:

**For Aarambh (long-history, valuation-determinant, ✓):**
`TERM_SPREAD_IN`, `TERM_SPREAD_US`, `REAL_REPO`, `IN_REAL_10Y`, `US_IN_10Y_DIFF`,
`INFLATION_MOM`, `BREADTH_MOM` — **7 sheet-derived, stationary, full-history.**
(These are the genuine macro determinants of an equity-valuation level.)

**For Nirnay/MMR (stock-level, can use the Y panel):** the 7 above **plus**
`CREDIT_HY_IG`, `DXY_MOM`, `GROWTH_FEAR`, `OIL_SHOCK`, `USDINR_MOM`, `RISK_ON_OFF`
— **13 total**, then the existing causal-PCA gate compresses to ~8 factors.

**Deliberately excluded from Aarambh:** the ⊘ valuation-ratio features
(`EQ_RISK_PREMIUM`, `IMPLIED_ROE`) — they embed PE and would re-introduce the
tautology that dropping PB/DY removed.

## Honest expectation
This is *correct* feature engineering — spreads/real-rates/credit are the right
primitives, and they have a fair shot at sharpening MMR and the convergence
readout. But PE-*level* forecasting has been falsified; richer features rotate
the same hard target. The non-overlapping signal-performance test and the
directional-convergence test remain the verdict, not a presumption of edge.

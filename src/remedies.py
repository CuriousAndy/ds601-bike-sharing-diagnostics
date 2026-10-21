"""Exclusion rules and the remedied model.

Owner: Ma Tianjiao. The rules are written before anyone looks at the test set and are
applied unchanged to all 731 days (training, CV folds and the test audit).

Rules may use predictors and the external event table only, never cnt, residuals or
anything computed from the test period. Every threshold below is a frozen constant that was
derived from the training set (see notebooks/03_influence_remedies.ipynb, section 1).
"""
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

# ---------------------------------------------------------------------------------------------
# Frozen rule constants (C1). Change only after telling the group leader.
# ---------------------------------------------------------------------------------------------

# D1 (data error), atemp part: reference line atemp = a + b * temp, fitted by OLS on the 584
# training days only. Scale = 1.4826 * MAD of the training residuals (a robust standard deviation).
# temp and atemp use different Celsius conversions, so the gap is measured against this fitted
# line, not as a raw difference.
ATEMP_LINE_INTERCEPT = 0.034592
ATEMP_LINE_SLOPE = 0.888273
ATEMP_RESID_SCALE = 0.013561
ATEMP_GAP_Z = 10.0  # flag a day when |robust z| > ATEMP_GAP_Z; training maximum is about 5.0

# Switch for the atemp part of D1. Decide in C1b whether it stays on if the remedied model drops atemp.
USE_ATEMP_RULE = True

# Event types (column `type` of src/external_events.csv, agreed in the group on 21 Oct).
# Only these two types exclude a day; every other type is kept and only flagged.
EXCLUDING_TYPES = {
    "P1_hurricane_storm": "hurricane_storm",   # named tropical storm / hurricane hitting DC (NOAA / NWS)
    "P2_federal_closure": "federal_closure",   # federal government closed all day for weather (OPM)
}
FLAG_ONLY_TYPES = ("snow", "early_dismissal", "event", "holiday", "other")

RULE_IDS = ("D1_hum_zero", "D1_atemp_gap", "P1_hurricane_storm", "P2_federal_closure")


def atemp_reference(train):
    """Fit the D1 reference line on the training set and return (intercept, slope, scale).

    Used once in notebook 03 to derive the constants above; exclusion_mask() never refits.
    """
    slope, intercept = np.polyfit(train["temp"], train["atemp"], 1)
    resid = train["atemp"] - (intercept + slope * train["temp"])
    scale = 1.4826 * np.median(np.abs(resid - np.median(resid)))
    return intercept, slope, scale


def atemp_gap_z(df):
    """Robust z-score of atemp against the frozen training reference line."""
    expected = ATEMP_LINE_INTERCEPT + ATEMP_LINE_SLOPE * df["temp"]
    return (df["atemp"] - expected) / ATEMP_RESID_SCALE


def _events_by_date(events, types):
    """{date: 'event (type); ...'} for rows of the given types."""
    if events is None or len(events) == 0:
        return {}
    ev = events.copy()
    ev["date"] = pd.to_datetime(ev["date"]).dt.normalize()
    ev["type"] = ev["type"].astype(str).str.strip().str.lower()
    ev = ev[ev["type"].isin(types)]
    label = ev["event"].astype(str) + " (" + ev["type"] + ")"
    return label.groupby(ev["date"]).apply(lambda s: "; ".join(s)).to_dict()


def exclusion_mask(df, events, use_atemp_rule=None):
    """Return a DataFrame indexed like df: excluded (bool), rule, reason, flag.

    Rules may use predictors and the external event table only, never cnt or residuals.
    A day that hits several rules lists all of them, separated by '+'. `flag` lists event-table
    entries of the kept types (snow, early dismissal, ...) for the adjudication table.
    """
    if use_atemp_rule is None:
        use_atemp_rule = USE_ATEMP_RULE
    dates = pd.to_datetime(df["dteday"]).dt.normalize()
    empty = pd.Series("", index=df.index)

    hits, reasons = {}, {}

    # D1, humidity part: relative humidity of exactly 0 is physically impossible outdoors.
    hits["D1_hum_zero"] = df["hum"] <= 0
    reasons["D1_hum_zero"] = pd.Series("hum = 0 (impossible value)", index=df.index)

    # D1, atemp part: feeling temperature inconsistent with air temperature.
    if use_atemp_rule:
        z = atemp_gap_z(df)
        hits["D1_atemp_gap"] = z.abs() > ATEMP_GAP_Z
        reasons["D1_atemp_gap"] = z.map(
            lambda v: f"atemp off the temp line by {v:+.1f} robust SD (limit {ATEMP_GAP_Z:g})"
        )
    else:
        hits["D1_atemp_gap"] = pd.Series(False, index=df.index)
        reasons["D1_atemp_gap"] = empty

    # P1 / P2: event types that mark a different population.
    for rid, etype in EXCLUDING_TYPES.items():
        found = _events_by_date(events, {etype})
        hits[rid] = dates.isin(list(found))
        reasons[rid] = dates.map(lambda d, f=found: f"external record: {f.get(d, '')}")

    hit_frame = pd.DataFrame(hits)[list(RULE_IDS)]
    out = pd.DataFrame(index=df.index)
    out["excluded"] = hit_frame.any(axis=1)
    out["rule"] = hit_frame.apply(lambda r: "+".join(c for c in RULE_IDS if r[c]), axis=1)
    out["reason"] = [
        "; ".join(reasons[c].loc[i] for c in RULE_IDS if hit_frame.at[i, c])
        for i in df.index
    ]
    flags = _events_by_date(events, set(FLAG_ONLY_TYPES))
    out["flag"] = dates.map(lambda d: flags.get(d, "")).values
    return out


def keep_rows(df, events, use_atemp_rule=None):
    """Convenience: rows of df that survive exclusion_mask()."""
    return df.loc[~exclusion_mask(df, events, use_atemp_rule)["excluded"]]


# ---------------------------------------------------------------------------------------------
# Remedy helpers (C2)
# ---------------------------------------------------------------------------------------------

def peak_temperature(res, var="temp"):
    """Turning point -b1 / (2 b2) of a quadratic in `var` (normalised scale)."""
    b1 = res.params[var]
    b2 = res.params[f"I({var} ** 2)"]
    return -b1 / (2 * b2)


def wls_weights(formula, df):
    """Two-step WLS weights: regress |OLS residual| on fitted, weight = 1 / sd_hat**2."""
    ols = smf.ols(formula, df).fit()
    aux = pd.DataFrame({"abs_resid": ols.resid.abs(), "fitted": ols.fittedvalues})
    sd_hat = smf.ols("abs_resid ~ fitted", aux).fit().fittedvalues
    sd_hat = sd_hat.clip(lower=sd_hat[sd_hat > 0].min() if (sd_hat > 0).any() else 1.0)
    return 1.0 / sd_hat ** 2


def fit_wls(formula, df):
    """Fit the two-step WLS model; returns the statsmodels result."""
    return smf.wls(formula, df, weights=wls_weights(formula, df)).fit()


# Frozen in C1b. Fill in the final choice, e.g.
# "cnt ~ yr + C(mnth) + holiday + C(weekday) + C(weathersit) + temp + I(temp ** 2) + hum + windspeed"
# and record whether it is OLS, WLS, log(cnt) or OLS with HC3 standard errors.
REMEDIED_FORMULA = None  # frozen in C1b; tell the leader before changing it afterwards
REMEDIED_METHOD = None   # one of "ols", "ols_hc3", "wls", "log"

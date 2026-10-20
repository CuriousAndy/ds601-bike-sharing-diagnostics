"""Diagnostics shared by the base and the remedied model.

Owner: Su Liangyu. Every function takes a fitted statsmodels OLS result `res`.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.diagnostic import het_breuschpagan


def _date_labels(dates):
    """Turn whatever date column we get into 'YYYY-MM-DD' strings."""
    return pd.to_datetime(pd.Series(dates).to_numpy()).strftime("%Y-%m-%d").to_numpy()


def _save(fig, save_as):
    if save_as is not None:
        fig.savefig(save_as, dpi=150, bbox_inches="tight")

def _date_of_row(res, i):
    """Look up dteday for row i from the data the model was fitted on."""
    frame = getattr(res.model.data, "frame", None)
    if frame is not None and "dteday" in frame:
        return pd.Timestamp(frame.loc[res.resid.index[i], "dteday"]).date()
    return res.resid.index[i]


def diagnostic_panel(res, dates, title, save_as=None):
    """2x2 panel: residuals vs fitted, normal Q-Q, scale-location, residuals vs leverage
    (Cook's D contours at 0.5 and 1). Labels the 3 most extreme points by date."""
    infl = res.get_influence()
    fitted = np.asarray(res.fittedvalues)
    resid = np.asarray(res.resid)
    stud = infl.resid_studentized_internal
    lev = infl.hat_matrix_diag
    cooks = infl.cooks_distance[0]
    p = len(res.params)
    labels = _date_labels(dates)
    top = np.argsort(cooks)[-3:]          # the 3 days with the largest Cook's D

    fig, axes = plt.subplots(2, 2, figsize=(11, 9))

    # 1. Residuals vs fitted -> linearity (a curved red line = missed non-linearity)
    ax = axes[0, 0]
    ax.scatter(fitted, resid, s=10, alpha=0.6)
    low = sm.nonparametric.lowess(resid, fitted)
    ax.plot(low[:, 0], low[:, 1], color="red")
    ax.axhline(0, color="grey", ls="--")
    for i in top:
        ax.annotate(labels[i], (fitted[i], resid[i]), fontsize=8)
    ax.set(xlabel="Fitted values", ylabel="Residuals", title="Residuals vs fitted")

    # 2. Normal Q-Q -> normality of errors (points leaving the line at the ends = heavy tails)
    sm.qqplot(stud, line="45", ax=axes[0, 1], markersize=3)
    axes[0, 1].set_title("Normal Q-Q (standardised residuals)")

    # 3. Scale-location -> constant variance (rising red line = heteroscedasticity)
    ax = axes[1, 0]
    sq = np.sqrt(np.abs(stud))
    ax.scatter(fitted, sq, s=10, alpha=0.6)
    low = sm.nonparametric.lowess(sq, fitted)
    ax.plot(low[:, 0], low[:, 1], color="red")
    for i in top:
        ax.annotate(labels[i], (fitted[i], sq[i]), fontsize=8)
    ax.set(xlabel="Fitted values", ylabel="sqrt(|standardised residual|)",
           title="Scale-location")

    # 4. Residuals vs leverage with Cook's D contours -> influential points
    ax = axes[1, 1]
    ax.scatter(lev, stud, s=10, alpha=0.6)
    h = np.linspace(max(lev.min(), 1e-3), lev.max() * 1.05, 200)
    for D, ls in [(0.5, "--"), (1.0, ":")]:
        bound = np.sqrt(D * p * (1 - h) / h)
        ax.plot(h, bound, color="red", ls=ls, label=f"Cook's D = {D}")
        ax.plot(h, -bound, color="red", ls=ls)
    ax.set_ylim(stud.min() - 1, stud.max() + 1)
    for i in top:
        ax.annotate(labels[i], (lev[i], stud[i]), fontsize=8)
    ax.set(xlabel="Leverage h", ylabel="Standardised residuals",
           title="Residuals vs leverage")
    ax.legend(fontsize=8)

    fig.suptitle(title)
    fig.tight_layout()
    _save(fig, save_as)
    return fig


def av_plots(res, variables, save_as=None):
    """Added-variable plots for the given predictors."""
    variables = list(variables)
    fig = plt.figure(figsize=(5 * len(variables), 4.5))
    sm.graphics.plot_partregress_grid(res, exog_idx=variables, fig=fig)
    fig.tight_layout()
    _save(fig, save_as)
    return fig


def vif_table(res):
    """VIF for every design-matrix column except the intercept, largest first."""
    X = np.asarray(res.model.exog)
    names = list(res.model.exog_names)
    rows = [
        (name, variance_inflation_factor(X, i))
        for i, name in enumerate(names)
        if name not in ("const", "Intercept")
    ]
    return (pd.DataFrame(rows, columns=["variable", "VIF"])
              .sort_values("VIF", ascending=False)
              .reset_index(drop=True))


def influence_table(res, dates):
    """One row per day: date, cnt, fitted, studentized residual, leverage h, Cook's D, DFFITS,
    plus 0/1 flags for h > 2p/n, D > 4/n and |DFFITS| > 2*sqrt(p/n) (p includes the intercept)."""
    infl = res.get_influence()
    n, p = int(res.nobs), len(res.params)
    out = pd.DataFrame({
        "date": pd.to_datetime(pd.Series(dates).to_numpy()),
        "cnt": np.asarray(res.model.endog),
        "fitted": np.asarray(res.fittedvalues),
        "stud_resid": infl.resid_studentized_external,
        "h": infl.hat_matrix_diag,
        "cooks_d": infl.cooks_distance[0],
        "dffits": infl.dffits[0],
    })
    out["flag_h"] = (out["h"] > 2 * p / n).astype(int)
    out["flag_cooks"] = (out["cooks_d"] > 4 / n).astype(int)
    out["flag_dffits"] = (out["dffits"].abs() > 2 * np.sqrt(p / n)).astype(int)
    return out


def summary_metrics(res):
    """dict: n, p, R2, adj R2, max Cook's D and its date, max VIF, Durbin-Watson, Breusch-Pagan p."""
    cooks = res.get_influence().cooks_distance[0]
    i = int(np.argmax(cooks))
    vif = vif_table(res)
    return {
        "n": int(res.nobs),
        "p": len(res.params),
        "R2": res.rsquared,
        "adj_R2": res.rsquared_adj,
        "max_cooks_d": cooks[i],
        "max_cooks_date": _date_of_row(res, i),
        "max_vif": vif["VIF"].max() if len(vif) else np.nan,
        "durbin_watson": durbin_watson(res.resid),
        "breusch_pagan_p": het_breuschpagan(res.resid, res.model.exog)[1],
    }
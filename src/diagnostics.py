"""Diagnostics shared by the base and the remedied model.

Owner: Su Liangyu. Every function takes a fitted statsmodels OLS result `res`.
"""


def diagnostic_panel(res, dates, title, save_as=None):
    """2x2 panel: residuals vs fitted, normal Q-Q, scale-location, residuals vs leverage
    (Cook's D contours at 0.5 and 1). Labels the 3 most extreme points by date."""
    raise NotImplementedError


def av_plots(res, variables, save_as=None):
    """Added-variable plots for the given predictors."""
    raise NotImplementedError


def vif_table(res):
    """VIF for every design-matrix column except the intercept, largest first."""
    raise NotImplementedError


def influence_table(res, dates):
    """One row per day: date, cnt, fitted, studentized residual, leverage h, Cook's D, DFFITS,
    plus 0/1 flags for h > 2p/n, D > 4/n and |DFFITS| > 2*sqrt(p/n) (p includes the intercept)."""
    raise NotImplementedError


def summary_metrics(res):
    """dict: n, p, R2, adj R2, max Cook's D and its date, max VIF, Durbin-Watson, Breusch-Pagan p."""
    raise NotImplementedError

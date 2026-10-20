"""Cross-validation used for every model comparison (metric: RMSE in rentals per day).

Owner: Zhang Enhao.

Both functions take the same arguments, so any candidate model can be scored either way:
    formula   patsy formula on the original scale, e.g. "cnt ~ yr + C(mnth) + temp"
    exclude   function df -> boolean Series (True = drop the day), e.g. remedies.exclusion_mask;
              applied the same way to every training and validation block
    log_y     fit log(cnt) instead of cnt; predictions are back-transformed with exp before
              the RMSE, so the error stays in rentals per day
    weights   function fit_df -> array of WLS weights, re-estimated inside every training block
"""
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from sklearn.model_selection import KFold

# The first fold must contain every month (so all of 2011) AND some 2012 days: trained on 2011
# alone, yr is constant, its effect cannot be estimated and fold 1 misses the whole 2012 jump.
FIRST_TRAIN_END = "2012-01-31"
N_BLOCKS = 3                    # 2012-02-01 .. 2012-08-06 = 188 days -> blocks of 63 / 63 / 62


def _excluded(exclude, df):
    """Boolean Series aligned with df: True for days the exclusion rule drops."""
    if exclude is None:
        return pd.Series(False, index=df.index)
    out = exclude(df)
    if isinstance(out, pd.DataFrame):
        out = out["excluded"]
    return pd.Series(np.asarray(out, dtype=bool), index=df.index)


def _fit_predict(formula, fit_df, pred_df, log_y=False, weights=None):
    """Fit on fit_df, return predictions for pred_df on the original cnt scale."""
    lhs, rhs = formula.split("~", 1)
    outcome = lhs.strip()
    f = f"np.log({outcome}) ~{rhs}" if log_y else formula
    if weights is None:
        res = smf.ols(f, fit_df).fit()
    else:
        res = smf.wls(f, fit_df, weights=weights(fit_df)).fit()
    pred = np.asarray(res.predict(pred_df))
    return np.exp(pred) if log_y else pred


def _rmse(y, pred):
    return float(np.sqrt(np.mean((np.asarray(y) - pred) ** 2)))


def forward_chain_cv(formula, train, exclude=None, log_y=False, weights=None,
                     first_train_end=FIRST_TRAIN_END, n_blocks=N_BLOCKS):
    """Expanding-window CV in time order.

    Fold k trains on every day before validation block k and validates on block k.
    Returns one row per fold: training end, block dates, months covered, n, RMSE.
    """
    df = train.sort_values("dteday").reset_index(drop=True)
    keep = ~_excluded(exclude, df)
    outcome = formula.split("~", 1)[0].strip()

    rest = df.index[df["dteday"] > pd.Timestamp(first_train_end)]
    blocks = np.array_split(rest, n_blocks)

    rows = []
    for k, block in enumerate(blocks, start=1):
        start = df.loc[block[0], "dteday"]
        fit_df = df[(df["dteday"] < start) & keep]
        val_df = df.loc[block][keep.loc[block]]
        assert fit_df["yr"].nunique() == 2, (
            f"fold {k}: training block has only one year, so the yr effect cannot be estimated"
        )
        pred = _fit_predict(formula, fit_df, val_df, log_y, weights)
        rows.append({
            "fold": k,
            "train_end": fit_df["dteday"].max().date(),
            "val_start": val_df["dteday"].min().date(),
            "val_end": val_df["dteday"].max().date(),
            "val_months": f"{val_df['dteday'].min():%b}-{val_df['dteday'].max():%b}",
            "n_train": len(fit_df),
            "n_val": len(val_df),
            "rmse": _rmse(val_df[outcome], pred),
        })
    return pd.DataFrame(rows)


def shuffled_kfold_cv(formula, train, k=5, seed=601, exclude=None, log_y=False, weights=None):
    """Ordinary shuffled k-fold CV, kept only for the comparison with forward chaining.

    Neighbouring days land in both the training and the validation folds, so with
    autocorrelated errors this estimate is expected to be optimistic.
    """
    df = train.reset_index(drop=True)
    df = df[~_excluded(exclude, df)].reset_index(drop=True)
    outcome = formula.split("~", 1)[0].strip()

    rows = []
    for fold, (fit_idx, val_idx) in enumerate(
            KFold(n_splits=k, shuffle=True, random_state=seed).split(df), start=1):
        fit_df, val_df = df.iloc[fit_idx], df.iloc[val_idx]
        pred = _fit_predict(formula, fit_df, val_df, log_y, weights)
        rows.append({"fold": fold, "n_train": len(fit_df), "n_val": len(val_df),
                     "rmse": _rmse(val_df[outcome], pred)})
    return pd.DataFrame(rows)


def cv_summary(folds):
    """Mean fold RMSE and its standard error (for the one-standard-error rule)."""
    r = folds["rmse"]
    return {"mean_rmse": float(r.mean()), "se": float(r.std(ddof=1) / np.sqrt(len(r)))}

"""Cross-validation used for every model comparison (metric: RMSE in rentals per day).

Owner: Zhang Enhao. The fold design is finalised after the Week 8 class on 23 Oct.
"""


def forward_chain_cv(formula, train, exclude=None, log_y=False, weights=None):
    """Expanding-window CV in time order. The first fold trains on all of 2011.

    exclude: function df -> boolean Series (e.g. remedies.exclusion_mask), applied the same way
    to every training and validation block. Returns one row per fold: block dates, months, n, RMSE.
    """
    raise NotImplementedError


def shuffled_kfold_cv(formula, train, k=5, seed=601, exclude=None, log_y=False, weights=None):
    """Ordinary shuffled k-fold CV, kept only for the comparison with forward chaining."""
    raise NotImplementedError

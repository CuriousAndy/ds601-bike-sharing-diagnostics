"""Exclusion rules and the remedied model.

Owner: Ma Tianjiao. The rules are written before anyone looks at the test set and are
applied unchanged to all 731 days (training, CV folds and the test audit).
"""

REMEDIED_FORMULA = None  # frozen in C1b; tell the leader before changing it afterwards


def exclusion_mask(df, events):
    """Return a DataFrame indexed like df: excluded (bool), rule, reason.

    Rules may use predictors and the external event table only, never cnt or residuals.
    """
    raise NotImplementedError

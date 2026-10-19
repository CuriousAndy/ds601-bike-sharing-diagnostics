"""Base model specification.

Owner: Su Liangyu.
"""

# Categorical variables enter as dummies. workingday is left out because it is an exact linear
# combination of the weekday dummies and holiday. Month vs season is chosen by forward-chaining CV (B2).
BASE_FORMULA = None

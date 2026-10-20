"""Base model specification.

Owner: Su Liangyu.
"""

# Categorical variables enter as dummies. workingday is left out because it is an exact linear
# combination of the weekday dummies and holiday. Month vs season is chosen by forward-chaining CV (B2).
BASE_FORMULA = (
    "cnt ~ yr + C(mnth) + holiday + C(weekday) + C(weathersit)"
    " + temp + atemp + hum + windspeed"
)
# Alternative compared in B2; whichever wins forward-chaining CV becomes BASE_FORMULA
SEASON_FORMULA = (
    "cnt ~ yr + C(season) + holiday + C(weekday) + C(weathersit)"
    " + temp + atemp + hum + windspeed"
)

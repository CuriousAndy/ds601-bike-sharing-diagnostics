"""Data loading, time-ordered split and unit conversion.

Owner: Jiang Liuru. Other members import from here and never re-split the data.
"""
from pathlib import Path
import urllib.request
import zipfile

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"  # git-ignored download cache
EVENTS_PATH = REPO_ROOT / "src" / "external_events.csv"  # small hand-built table, committed

UCI_ZIP_URL = "https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip"
TRAIN_END = "2012-08-06"  # last training day, inclusive -> 584 train / 147 test

OUTCOME = "cnt"
BASE_PREDICTORS = [
    "season", "yr", "mnth", "holiday", "weekday", "workingday",
    "weathersit", "temp", "atemp", "hum", "windspeed",
]
LEAKAGE = ["casual", "registered"]  # cnt = casual + registered: never a predictor or a grouping variable


def load_day():
    """Download the UCI zip into DATA_DIR (skip if present), read day.csv, parse dteday.

    Checks: 731 rows, no NaN, cnt == casual + registered on every row, no missing dates.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = DATA_DIR / "bike+sharing+dataset.zip"
    if not zip_path.exists():
        urllib.request.urlretrieve(UCI_ZIP_URL, zip_path)

    with zipfile.ZipFile(zip_path) as archive:
        with archive.open("day.csv") as day_file:
            df = pd.read_csv(day_file, parse_dates=["dteday"])

    assert df.shape == (731, 16), "day.csv must contain exactly 731 rows and 16 columns"
    assert not df.isna().any().any(), "day.csv must not contain missing values"
    assert (df["cnt"] == df["casual"] + df["registered"]).all(), (
        "cnt must equal casual + registered on every row"
    )
    expected_dates = pd.date_range("2011-01-01", "2012-12-31", freq="D")
    assert not df["dteday"].duplicated().any(), "dteday must not contain duplicates"
    assert df["dteday"].equals(pd.Series(expected_dates, name="dteday")), (
        "dteday must contain every day from 2011-01-01 through 2012-12-31"
    )
    return df.sort_values("dteday").reset_index(drop=True)


def time_split(df):
    """Return (train, test): train = dteday <= TRAIN_END. Asserts 584 / 147 rows."""
    train = df.loc[df["dteday"] <= pd.Timestamp(TRAIN_END)].copy().reset_index(drop=True)
    test = df.loc[df["dteday"] > pd.Timestamp(TRAIN_END)].copy().reset_index(drop=True)
    assert len(train) == 584, "training split must contain exactly 584 rows"
    assert len(test) == 147, "test split must contain exactly 147 rows"
    return train, test


def add_real_units(df):
    """Add temp_c, atemp_c, hum_pct, wind_kmh for plots and interpretation only.

    The temperature formula is decided in A1 (zip Readme vs UCI web page, checked against NOAA).
    """
    result = df.copy()
    result["temp_c"] = result["temp"] * 47 - 8
    result["atemp_c"] = result["atemp"] * 66 - 16
    result["hum_pct"] = result["hum"] * 100
    result["wind_kmh"] = result["windspeed"] * 67
    return result


def load_events():
    """Read EVENTS_PATH: one row per notable day in 2011-2012 (date, event, type, source_url).

    Built from external records only, never by looking at cnt.
    """
    raise NotImplementedError

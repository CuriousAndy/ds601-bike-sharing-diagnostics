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

    assert df.shape == (731, 16), (
        f"day.csv must contain exactly 731 rows and 16 columns; got shape {df.shape}"
    )
    assert not df.isna().any().any(), "day.csv must not contain missing values"
    assert (df["cnt"] == df["casual"] + df["registered"]).all(), (
        "cnt must equal casual + registered on every row"
    )
    assert df["dteday"].is_unique and pd.DatetimeIndex(
        df["dteday"].sort_values()
    ).equals(pd.date_range("2011-01-01", "2012-12-31", freq="D")), (
        "dteday must be unique and contain every day from 2011-01-01 through 2012-12-31"
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


EVENT_TYPES = {"hurricane_storm", "federal_closure", "snow", "early_dismissal", "event", "holiday", "other"}


def load_events():
    """Read EVENTS_PATH: notable days in 2011-2012 (date, event, type, source_url).

    One row per (date, type); a day can have several rows, e.g. Sandy is both
    hurricane_storm and federal_closure. Built from external records only, never by looking at cnt.
    """
    events = pd.read_csv(EVENTS_PATH, parse_dates=["date"])
    expected = ["date", "event", "type", "source_url"]
    assert list(events.columns) == expected, f"columns must be {expected}, got {list(events.columns)}"
    unknown = sorted(set(events["type"]) - EVENT_TYPES)
    assert not unknown, f"unknown event type(s) {unknown}; allowed: {sorted(EVENT_TYPES)}"
    assert not events.duplicated(["date", "type"]).any(), "duplicate (date, type) rows"
    assert events["date"].between("2011-01-01", "2012-12-31").all(), "every date must be in 2011-2012"
    assert events["source_url"].notna().all() and (events["source_url"].str.strip() != "").all(), \
        "every event needs a source_url"
    return events.sort_values(["date", "type"]).reset_index(drop=True)
"""Data loading, time-ordered split and unit conversion.

Owner: Jiang Liuru. Other members import from here and never re-split the data.
"""
from pathlib import Path

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
    raise NotImplementedError


def time_split(df):
    """Return (train, test): train = dteday <= TRAIN_END. Asserts 584 / 147 rows."""
    raise NotImplementedError


def add_real_units(df):
    """Add temp_c, atemp_c, hum_pct, wind_kmh for plots and interpretation only.

    The temperature formula is decided in A1 (zip Readme vs UCI web page, checked against NOAA).
    """
    raise NotImplementedError


def load_events():
    """Read EVENTS_PATH: one row per notable day in 2011-2012 (date, event, type, source_url).

    Built from external records only, never by looking at cnt.
    """
    raise NotImplementedError

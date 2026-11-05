import hashlib
import io
import json
import re
from pathlib import Path
import numpy as np
import pandas as pd

DATA = Path(__file__).parent / "data"
UA = {"User-Agent": "PriceWise/2.0 (student project; single file download)"}


class DataError(Exception):
    pass


def digest(b):
    return hashlib.md5(b).hexdigest()[:10]


def _numeric(s):
    return pd.to_numeric(s.astype(str).str.replace(",", "").str.strip().replace({".": np.nan, "": np.nan, "NA": np.nan, "nan": np.nan}), errors="coerce")


def _dates(s):
    s = s.astype(str).str.strip().str.replace(r"^(\d{4})M(\d{2})$", r"\1-\2", regex=True).str.replace(r"^(\d{4})Q([1-4])$", lambda m: f"{m.group(1)}-{int(m.group(2)) * 3 - 2:02d}", regex=True)
    return pd.to_datetime(s, errors="coerce")


def read_series(name, data):
    """Parse CSV/Excel bytes into a monthly pandas Series (price level)."""
    try:
        df = pd.read_excel(io.BytesIO(data)) if name.lower().endswith((".xlsx", ".xls")) else pd.read_csv(io.BytesIO(data), sep=None, engine="python")
    except Exception as e:
        raise DataError(f"Could not read the file ({e.__class__.__name__}). Use a CSV or Excel file with a date column and a price column.") from e
    if df.shape[1] < 2 or len(df) < 12:
        raise DataError("The file needs a date column, a price column and at least 12 rows.")
    cols = list(df.columns)
    dcol = next((c for c in cols if str(c).strip().lower() in ("date", "observation_date", "month", "period", "time")), None)
    if dcol is None:
        dcol = next((c for c in cols if _dates(df[c]).notna().mean() > 0.8), None)
    if dcol is None:
        raise DataError("No date column found. Name it 'date' (for example 2024-05-01).")
    others = [c for c in cols if c != dcol]
    vcol = next((c for c in others if str(c).strip().lower() in ("value", "price")), None) or next((c for c in others if _numeric(df[c]).notna().mean() > 0.8), None)
    if vcol is None:
        raise DataError("No price column found. Name it 'price' or 'value'.")
    s = pd.Series(_numeric(df[vcol]).values, index=_dates(df[dcol])).loc[lambda x: x.index.notna()].sort_index()
    s = s[~s.index.duplicated(keep="last")]
    if (s.dropna() <= 0).any():
        raise DataError("The file contains zero or negative prices.")
    gap = np.median(np.diff(s.index.values).astype("timedelta64[D]").astype(float)) if len(s) > 2 else 99
    if gap > 45:
        raise DataError("The file needs monthly, weekly or daily data. Yearly or quarterly files are too coarse.")
    m = s.resample("MS").mean()
    first, last = m.first_valid_index(), m.last_valid_index()
    m = m.loc[first:last]
    filled = int(m.isna().sum())
    if filled > 0.2 * len(m):
        raise DataError(f"Too many missing months ({filled} of {len(m)}).")
    m = m.interpolate(limit=6, limit_area="inside")
    if m.isna().any():
        raise DataError("Long gaps of missing prices in the file.")
    m.attrs.update(filled=filled, weekly=gap <= 10)
    return m


def fetch(url, timeout=20):
    try:
        import requests
        r = requests.get(url, headers=UA, timeout=timeout)
        r.raise_for_status()
        return r.content
    except Exception as e:
        raise DataError(f"Download failed ({e.__class__.__name__}). Use the link to download the file, then upload it here.") from e


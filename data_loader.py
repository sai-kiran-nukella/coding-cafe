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


def saved(key):
    f = DATA / f"{key}.csv"
    if not f.exists():
        return None
    meta = DATA / f"{key}.meta.json"
    return {"name": f.name, "bytes": f.read_bytes(), "meta": json.loads(meta.read_text()) if meta.exists() else {}}


def save(key, data, how, url=""):
    DATA.mkdir(exist_ok=True)
    (DATA / f"{key}.csv").write_bytes(data)
    (DATA / f"{key}.meta.json").write_text(json.dumps({"fetched": pd.Timestamp.today().strftime("%Y-%m-%d"), "how": how, "source_url": url}))


def normalise_upload(name, data):
    """Uploaded file (any supported layout) to a clean date,value CSV so it can be reused later."""
    s = read_series(name, data)
    return s.rename("value").rename_axis("date").to_csv().encode()


def build_market(price_name, price_bytes, cpi_bytes=None, today=None):
    p = read_series(price_name, price_bytes)
    info = {"first": p.index[0], "last": p.index[-1], "months": len(p), "filled": p.attrs.get("filled", 0), "weekly": p.attrs.get("weekly", False), "inflation": False}
    today = pd.Timestamp(today or pd.Timestamp.today())
    info["stale_months"] = (today.year - p.index[-1].year) * 12 + today.month - p.index[-1].month
    if len(p) < 120:
        raise DataError(f"Only {len(p)} months of data. At least 10 years (120 months) are needed to learn the market's patterns.")
    cpi = None
    if cpi_bytes is not None:
        c = read_series("cpi.csv", cpi_bytes)
        c = c.reindex(pd.date_range(min(c.index[0], p.index[0]), max(c.index[-1], p.index[-1]), freq="MS"))
        gm = np.log(c.dropna()).diff().tail(6).mean()
        for i in range(len(c)):
            if np.isnan(c.iloc[i]) and i > 0 and c.index[i] > c.dropna().index[-1]:
                c.iloc[i] = c.iloc[i - 1] * np.exp(gm)
        c = c.interpolate(limit_area="inside")
        start = max(p.index[0], c.first_valid_index())
        p, c = p.loc[start:], c.loc[start:p.index[-1]]
        if len(p) >= 120 and not c.isna().any():
            cpi, info["inflation"], info["cpi_last"] = c, True, c.dropna().index[-1]
    return {"price": p, "cpi": cpi, "info": info}

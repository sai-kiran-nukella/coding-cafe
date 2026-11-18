import io
import hashlib
import re
import numpy as np
import pandas as pd
from sources import city_name

KEY = ['city','product','market','variety','grade','basis','unit','source']
COLUMNS = KEY + ['date','price','retrieved_at']
FACTORS = {'kg':('kg',1), 'quintal':('kg',100), 'qtl':('kg',100), 'tonne':('kg',1000), 'litre':('litre',1), 'liter':('litre',1), 'dozen':('dozen',1)}
class DataError(ValueError): pass

def today():
    return pd.Timestamp.now(tz='Asia/Kolkata').normalize().tz_localize(None)

def load_file(name, content, source, unit='quintal', basis='wholesale', declared_city=None):
    if not source.strip(): raise DataError('Enter the publisher or source URL.')
    if len(content)>8_000_000: raise DataError('Choose a file smaller than 8 MB.')
    try:
        frame = pd.read_excel(io.BytesIO(content),dtype=str) if name.lower().endswith('.xlsx') else pd.read_csv(io.BytesIO(content),dtype=str)
    except Exception as e: raise DataError('Use a readable CSV or XLSX file.') from e
    frame.columns = [re.sub('[^a-z0-9]','',str(c).lower()) for c in frame.columns]
    aliases = {'arrivaldate':'date','commodity':'product','modalprice':'price','reportedprice':'price','centre':'city'}
    frame = frame.rename(columns=aliases)
    if frame.columns.duplicated().any(): raise DataError('Multiple columns map to the same field. Keep one date and one price column.')
    if not {'date','price','product','market'}.issubset(frame): raise DataError('Required columns: date, price, product, market. Commodity/modal_price aliases are accepted.')
    if 'city' not in frame:
        if not declared_city: raise DataError('Add a city column, or explicitly assign this single-city file below.')
        frame['city']=declared_city
    else:
        # Never override an existing city column using the selected sidebar city.
        frame['city']=frame['city'].map(city_name)
    frame['city']=frame['city'].map(city_name)
    for field, default in [('variety','Not specified'),('grade','Not specified'),('unit',unit),('basis',basis)]:
        if field not in frame: frame[field]=default
        frame[field]=frame[field].fillna(default).astype(str).str.strip()
    dates=frame['date'].fillna('').str.strip()
    iso=dates.str.fullmatch(r'\d{4}-\d{2}-\d{2}')
    indian=dates.str.fullmatch(r'\d{2}[/-]\d{2}[/-]\d{4}')
    parsed=pd.Series(pd.NaT,index=frame.index,dtype='datetime64[ns]')
    parsed.loc[iso]=pd.to_datetime(dates[iso],format='%Y-%m-%d',errors='coerce')
    parsed.loc[indian]=pd.to_datetime(dates[indian].str.replace('/','-'),format='%d-%m-%Y',errors='coerce')
    frame['date']=parsed
    frame['price']=pd.to_numeric(frame['price'].astype(str).str.replace(',','').str.replace('₹','').str.strip(),errors='coerce')
    frame['unit']=frame['unit'].str.lower()
    for c in ['product','market']:
        frame[c]=frame[c].fillna('').astype(str).str.strip()
    valid=frame.city.notna() & frame.date.notna() & (frame.date<=today()) & np.isfinite(frame.price) & (frame.price>0) & frame.unit.isin(FACTORS) & frame.basis.isin(['retail','wholesale']) & frame['product'].ne('') & frame.market.ne('')
    rejected=int((~valid).sum()); frame=frame.loc[valid].copy()
    if frame.empty: raise DataError('No valid records for the six supported cities. Check dates, names, prices and units.')
    frame['price']=[p/FACTORS[u][1] for p,u in zip(frame.price,frame.unit)]
    frame['unit']=[FACTORS[u][0] for u in frame.unit]
    frame['source']=source.strip(); frame['retrieved_at']=pd.Timestamp.now(tz='UTC').isoformat()
    frame=frame[COLUMNS]
    conflict=frame.duplicated(KEY+['date'],keep=False)
    if conflict.any(): raise DataError('Duplicate dates in a series. Resolve duplicate or revised observations before importing.')
    return frame.sort_values('date'), {'accepted':len(frame),'rejected':rejected,'sha256':hashlib.sha256(content).hexdigest()}

def quality(group, now=None):
    g=group.sort_values('date'); dates=pd.to_datetime(g.date); n=len(g)
    span=int((dates.iloc[-1]-dates.iloc[0]).days) if n else 0
    business=len(pd.bdate_range(dates.iloc[0],dates.iloc[-1])) if n else 1
    coverage=min(1,n/max(business,1)); gap=int(dates.diff().dt.days.max()) if n>1 else 0
    reasons=[]
    if n<60: reasons.append('at least 60 observations')
    if span<90: reasons.append('at least 90 calendar days')
    if coverage<.65: reasons.append('at least 65% weekday coverage')
    if gap>14: reasons.append('no reporting gap longer than 14 days')
    age=int(((pd.Timestamp(now).normalize() if now is not None else today())-dates.iloc[-1]).days) if n else 999
    return {'eligible':not reasons,'observations':n,'span_days':span,'coverage':coverage,'max_gap_days':gap,'age_days':age,'reason':'; '.join(reasons),'latest':dates.iloc[-1]}

def catalogue(frame,city):
    rows=[]
    if frame.empty:return pd.DataFrame()
    for key,g in frame[frame.city==city].groupby(KEY,dropna=False):
        q=quality(g); rows.append({**dict(zip(KEY,key)),**q})
    return pd.DataFrame(rows)

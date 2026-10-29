import pickle
from dataclasses import dataclass
from pathlib import Path
from fair_price_net import FairPriceModel
from simulator import GOODS, training_paths
from trend_net import TrendModel

VERSION = 4
CACHE = Path(__file__).parent / ".cache"


@dataclass
class Bundle:
    fair: FairPriceModel
    trend: TrendModel


def train(good):
    df = training_paths(good, 48, 60, 1000 + list(GOODS).index(good))
    fair = FairPriceModel(good).fit(df)
    return Bundle(fair, TrendModel(good).fit(df, fair))


def load_or_train(good, force=False):
    f = CACHE / f"{good.replace(' ', '_')}_v{VERSION}.pkl"
    if f.exists() and not force:
        try:
            return pickle.loads(f.read_bytes())
        except Exception:
            pass
    b = train(good)
    try:
        CACHE.mkdir(exist_ok=True)
        f.write_bytes(pickle.dumps(b))
    except OSError:
        pass
    return b

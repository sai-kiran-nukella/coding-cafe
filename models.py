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




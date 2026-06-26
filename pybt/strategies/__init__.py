from .ma25_pullback import MA25Pullback
from .orb import ORB
from .vwap import VWAPReversion

# run.py から名前で引けるレジストリ
REGISTRY = {
    "orb": ORB,
    "ma25": MA25Pullback,
    "vwap": VWAPReversion,
}

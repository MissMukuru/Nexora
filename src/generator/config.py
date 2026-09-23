#controls teh overall scale and resproducibility of the generator
from dataclasses import dataclass, asdict
from pathlib import Path
import hashlib
import numpy as np

REGIONS = np.array(["Nairobi", "Kiambu", "Nakuru", "Eldoret", "Mombasa", "Kisumu"])
SEGMENTS = np.array(["Micro", "SMB", "Midmarket", "Enterprise"])
CATEGORIES = np.array(["Software", "Hardware", "Consulting", "Training", "Support"])
CHANNELS = np.array(["Search", "Social", "Email", "Partner", "Events", "Direct"])
START = np.datetime64("2025-01-01", "D")
END = np.datetime64("2026-12-31", "D")
DAYS = 730


def rng(name, *keys, seed=42):
    """Stable namespaces, independent of Python hash randomization and call ordering."""
    digest = hashlib.blake2b(
        "|".join(map(str, (name, *keys))).encode(), digest_size=16
    ).digest()
    return np.random.Generator(
        np.random.PCG64(
            np.random.SeedSequence([seed, *np.frombuffer(digest, dtype="<u4").tolist()])
        )
    )


@dataclass(frozen=True)
class Config:#defines teh record counts and random seed
    output: str = "data"
    seed: int = 42
    customers: int = 65000
    products: int = 650
    employees: int = 2600
    transactions: int = 6500000
    interactions: int = 2600000
    subscriptions: int = 150000
    campaigns: int = 6500
    usage: int = 6500000
    pipeline: int = 650000
    support: int = 1300000
    episodes: int = 220
    scale: float = 1.0

    def count(self, name):
        # Explicit developer fixture option; production defaults never scaled down.
        return max(1, round(getattr(self, name) * self.scale))

    def stream(self, name, *keys):
        return rng(name, *keys, seed=self.seed)

    def metadata(self):
        return asdict(self)

    @property
    def root(self):
        return Path(self.output)


def date(day):
    return START + np.asarray(day).astype("timedelta64[D]") #converts day offsets into dates


def times(day, n, r):
    # Business-hour mixture, with overnight online activity.
    seconds = np.clip(r.normal(13 * 3600, 3 * 3600, n), 0, 86399).astype("int64")
    overnight = r.random(n) < 0.12
    seconds[overnight] = r.integers(0, 86400, overnight.sum())
    return (
        date(day).astype("datetime64[s]") + seconds.astype("timedelta64[s]")
    ).astype("datetime64[ms]")


def money(x):
    return np.round(x, 2)

#this is the probabilistic calcumations
def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -30, 30)))

from dataclasses import dataclass, field
import numpy as np
import polars as pl
from .config import *


@dataclass
class World:
    cfg: Config
    frames: dict = field(default_factory=dict)
    ids: dict = field(default_factory=dict)

    def r(self, name, *keys): #gets a deterministc random stream
        return self.cfg.stream(name, *keys)

    def next_ids(self, name, n): #generates unique ids
        start = self.ids.get(name, 0) + 1
        self.ids[name] = start + n - 1
        return np.arange(start, start + n, dtype="int64")

    def init_state(self): #for customer behaviour
        n = self.cfg.count("customers")
        r = self.r("customer-process")
        self.engagement = r.beta(5, 2, n)
        self.credit = r.beta(2, 9, n)
        self.support_burden = np.zeros(n)
        self.satisfaction = np.full(n, 4.2)
        self.last_purchase = np.full(n, -1000)
        self.revenue = np.zeros(n)
        self.payment_friction = np.zeros(n)
        self.owned = np.zeros((n, 8), dtype="int32")
        self.owned_count = np.zeros(n, dtype="int32")
        self.churned = np.zeros(n, dtype=bool)
        self.support_updates = {}

    def customers_for(self, n, day, r, region=None, segment=None):
        eligible = self.acq <= day
        if region is not None:
            eligible &= self.cr == region
        if segment is not None:
            eligible &= self.cs == segment
        ix = np.flatnonzero(eligible)
        if not len(ix):
            raise ValueError(
                f"No eligible customer in target {region}/{segment} on {day}"
            )
        weights = (
            self.propensity[ix]
            * (0.2 + self.engagement[ix])
            * (1 - 0.45 * self.churned[ix])
        )
        weights *= np.exp(
            -0.25 * self.support_burden[ix] - 0.2 * self.payment_friction[ix]
        )
        if hasattr(self, "ext"):
            reg = self.cr[ix]
            weights *= self.ext["market"][day, reg] * np.exp(
                0.018
                * (self.ext["confidence"][day, reg] - 60)
                * (1 + 0.3 * self.cs[ix])
                - 0.024 * (self.ext["inflation"][day, reg] - 5)
            )
        return r.choice(ix, n, p=weights / weights.sum())

    def remember(self, c, p):
        # Stable per-customer final row wins, no reliance on repeated advanced-index assignment.
        if len(c):
            _, rev = np.unique(c[::-1], return_index=True)
            ix = len(c) - 1 - rev
            cc = c[ix]
            slot = self.owned_count[cc] % 8
            self.owned[cc, slot] = p[ix] + 1
            self.owned_count[cc] += 1

    def owned_product(self, c, r):
        count = np.minimum(self.owned_count[c], 8)
        slot = (r.random(len(c)) * np.maximum(count, 1)).astype(int)
        p = self.owned[c, slot] - 1
        return p

    def employee(self, department, regions, r):
        out = np.empty(len(regions), dtype="int64")
        for region in np.unique(regions):
            pool = np.flatnonzero((self.ed == department) & (self.er == region))
            if not len(pool):
                pool = np.flatnonzero(self.ed == department)
            out[regions == region] = r.choice(pool, (regions == region).sum()) + 1
        return out

"""Private injection engine. Metadata never enters analytical records."""

from collections import defaultdict
import numpy as np
from .config import date

DOMAINS = [
    "demand",
    "discount",
    "support",
    "usage",
    "payments",
    "inventory",
    "operations",
    "subscriptions",
    "campaigns",
    "pipeline",
]


class Episodes:
    def __init__(self, w):
        r = w.r("episodes")
        self.items = []
        self.counts = defaultdict(lambda: defaultdict(int))
        self.active = {}
        for i in range(w.cfg.episodes):
            domain = DOMAINS[i % len(DOMAINS)]
            scope = r.choice(
                ["region", "customer", "product", "segment"], p=[0.35, 0.2, 0.3, 0.15]
            )
            if domain == "operations":
                scope = "region"
            if domain == "campaigns":
                scope = r.choice(["region", "product"])
            start = int(r.integers(2, 700))
            duration = int(
                r.choice([1, 3, 7, 14, 30, 60], p=[0.18, 0.13, 0.2, 0.23, 0.18, 0.08])
            )
            hours = bool(duration == 1 and r.random() < 0.6)
            start_hour = int(r.integers(7, 16)) if hours else 0
            severity = str(r.choice(["low", "moderate", "high"], p=[0.48, 0.37, 0.15]))
            placebo = bool(r.random() < 0.12)
            mag = {"low": 0.18, "moderate": 0.45, "high": 0.95}[severity]
            factor = float(np.exp(r.choice([-1, 1]) * mag)) if not placebo else 1.0
            bound = {
                "region": 6,
                "customer": w.cfg.count("customers"),
                "product": w.cfg.count("products"),
                "segment": 4,
            }[scope]
            self.items.append(
                dict(
                    episode_id=f"LAT-{i+1:03d}",
                    domain=domain,
                    scope=scope,
                    target=int(r.integers(bound)),
                    start=start,
                    end=min(729, start + duration - 1),
                    start_hour=start_hour,
                    end_hour=min(23, start_hour + 3) if hours else 24,
                    severity=severity,
                    factor=factor,
                    placebo=placebo,
                    interpretation=(
                        "Counterfactual no-op; candidate noise/seasonal window"
                        if placebo
                        else "Stochastic process intervention; not a unique analytical answer"
                    ),
                )
            )
        self.benchmarks = [
            dict(
                event_id="EVT-001",
                start="2026-03-15",
                end="2026-04-15",
                scope=["Nairobi"],
                mechanism="Capacity loss, absenteeism and downtime; propagates through backlog and support",
            ),
            dict(
                event_id="EVT-002",
                start="2026-04-01",
                end="2026-12-31",
                scope=["Software"],
                mechanism="1.12 price multiplier; heterogeneous demand elasticity and renewal response",
            ),
            dict(
                event_id="EVT-003",
                start="2026-05-01",
                end="2026-05-31",
                scope=["Marketing"],
                mechanism="Higher spend and raw leads, lower qualification and win rates",
            ),
            dict(
                event_id="EVT-004",
                start="2026-06-01",
                end="2026-06-30",
                scope=["Nakuru", "Eldoret"],
                mechanism="Fuel/inflation shock, confidence loss, nonlinear demand and expense response",
            ),
        ]

    def set_day(self, day):
        self.active = {
            d: [
                e
                for e in self.items
                if e["domain"] == d and e["start"] <= day <= e["end"]
            ]
            for d in DOMAINS
        }

    def factor(
        self,
        domain,
        day,
        region,
        customer=None,
        product=None,
        segment=None,
        ts=None,
        table=None,
    ):
        n = len(region)
        out = np.ones(n)
        mapping = dict(
            region=region, customer=customer, product=product, segment=segment
        )
        for e in self.active.get(domain, []):
            values = mapping[e["scope"]]
            if values is None:
                continue
            mask = values == e["target"]
            if ts is not None and e["end_hour"] != 24:
                h = ts.astype("datetime64[h]").astype("int64") % 24
                mask &= (h >= e["start_hour"]) & (h < e["end_hour"])
            elif e["end_hour"] != 24:
                # Daily aggregates carry only the fractional-day exposure.
                out[mask] *= (
                    1 + (e["factor"] - 1) * (e["end_hour"] - e["start_hour"]) / 24
                )
                self.counts[e["episode_id"]][table or domain] += int(mask.sum())
                continue
            out[mask] *= e["factor"]
            self.counts[e["episode_id"]][table or domain] += int(mask.sum())
        return np.clip(out, 0.15, 6.0)

    def benchmark(self, event, table, n):
        self.counts[event][table] += int(n)

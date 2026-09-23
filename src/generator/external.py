import numpy as np
import polars as pl
from .config import *


def generate(w):
    r = w.r("external")
    t = np.arange(DAYS)
    n = DAYS

    def ar(scale):
        x = np.zeros((n, 6))
        noise = r.normal(0, scale, (n, 6))
        for d in range(1, n):
            x[d] = 0.93 * x[d - 1] + noise[d]
        return x

    month = (date(t).astype("datetime64[M]").astype(int) % 12) + 1
    holiday = np.isin(
        np.datetime_as_string(date(t)),
        [
            "2025-01-01",
            "2025-04-18",
            "2025-04-21",
            "2025-05-01",
            "2025-06-01",
            "2025-10-10",
            "2025-10-20",
            "2025-12-12",
            "2025-12-25",
            "2025-12-26",
            "2026-01-01",
            "2026-04-03",
            "2026-04-06",
            "2026-05-01",
            "2026-06-01",
            "2026-10-10",
            "2026-10-20",
            "2026-12-12",
            "2026-12-25",
            "2026-12-26",
        ],
    )
    # Fictional holiday calendar: fixed civic dates plus Easter; not legal-calendar authority.
    fuel = 178 + 7 * np.sin(t[:, None] / 180) + np.array([0, 1, 3, 5, 2, 4]) + ar(0.45)
    inflation = 5.1 + 0.35 * np.sin(t[:, None] / 130) + ar(0.035)
    confidence = 61 + 4 * np.sin(t[:, None] / 90) + ar(0.6)
    shock = (t >= 516) & (t <= 545)
    fuel[shock, 2:4] += 29
    inflation[shock, 2:4] += 2.1
    confidence[shock, 2:4] -= 16
    rainfall = np.maximum(
        0,
        35
        + 28 * np.isin(month[:, None], [3, 4, 5, 10, 11])
        + ar(5)
        + r.gamma(2, 4, (n, 6)),
    )
    traffic = np.clip(
        0.52
        + np.array([0.25, 0.12, -0.02, -0.04, 0.1, -0.01])
        + ar(0.018)
        + 0.001 * rainfall,
        0.1,
        1.4,
    )
    fx = 129 + 2 * np.sin(t[:, None] / 200) + ar(0.13)
    market = np.clip(
        1 + 0.003 * (confidence - 60) - 0.014 * (inflation - 5) + ar(0.012), 0.5, 1.4
    )
    w.ext = dict(
        fuel=fuel,
        inflation=inflation,
        confidence=confidence,
        rainfall=rainfall,
        traffic=traffic,
        fx=fx,
        market=market,
        holiday=holiday,
    )
    w.frames["external_factors"] = pl.DataFrame(
        dict(
            date=np.repeat(date(t), 6),
            region=np.tile(REGIONS, n),
            fuel_price=fuel.ravel(),
            inflation_rate=inflation.ravel(),
            business_confidence=confidence.ravel(),
            exchange_rate=fx.ravel(),
            rainfall_index=rainfall.ravel(),
            traffic_index=traffic.ravel(),
            market_activity_index=market.ravel(),
            holiday_flag=np.repeat(holiday, 6),
        )
    )
    w.ep.benchmark("EVT-004", "external_factors", int(shock.sum() * 2))


def schedule(w, name, total):
    r = w.r("schedule", name)
    day = np.arange(DAYS)
    dates = date(day)
    dow = (dates.astype(int) + 3) % 7
    month = dates.astype("datetime64[M]").astype(int) % 12 + 1
    dom = (dates - dates.astype("datetime64[M]")).astype(int) + 1
    weights = (
        np.where(dow < 5, 1, 0.45)
        * (1 + 0.08 * np.cos(2 * np.pi * day / 365))
        * (1 + 0.10 * (dom >= 25))
    )
    weights *= (
        np.where(w.ext["holiday"], 0.6, 1)
        * (1 + 0.1 * (month % 3 == 0))
        * (1 + 0.00035 * day)
    )
    weights *= np.exp(r.normal(0, 0.09, DAYS))
    # Region aggregate confidence and market state influence the volume process.
    weights *= w.ext["market"].mean(axis=1)
    # Allocation conditions on a requested total, preserving realistic temporal shape.
    return r.multinomial(total, weights / weights.sum())

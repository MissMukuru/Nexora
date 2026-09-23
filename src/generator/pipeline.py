import numpy as np
import polars as pl
from .config import *


def initialize(w):
    r = w.r("pipeline")
    n = w.cfg.count("pipeline")
    a = w.campaign
    weights = np.sqrt(a["spend"]) * a["quality"] * (1 + 0.3 * a["overlap"])
    counts = r.multinomial(n, weights / weights.sum())
    campaign = np.repeat(np.arange(len(counts)), counts)
    c = np.empty(n, dtype=int)
    created = np.empty(n, dtype=int)
    offset = 0
    # Campaign-level loop, never a loop over individual opportunities.
    for i, k in enumerate(counts):
        if not k:
            continue
        ix = slice(offset, offset + k)
        offset += k
        c[ix] = w.customers_for(
            k, int(a["start"][i]), r, region=a["region"][i], segment=a["segment"][i]
        )
        created[ix] = r.integers(a["start"][i], a["end"][i] + 1, k)
    p = a["product"][campaign]
    region = w.cr[c]
    duration = np.clip(r.gamma(2.6, 10 + 8 * w.cs[c]), 3, 240).astype(int)
    close = created + duration
    wave = (created >= 485) & (created <= 515)
    win = sigmoid(
        -0.6
        + 0.45 * a["quality"][campaign]
        + 0.22 * w.cs[c]
        - 0.8 * wave
        + 0.012 * (w.ext["confidence"][np.minimum(close, 729), region] - 60)
    )
    for day in np.unique(created):
        ix = np.flatnonzero(created == day)
        w.ep.set_day(int(day))
        win[ix] *= w.ep.factor(
            "pipeline",
            int(day),
            region[ix],
            c[ix],
            p[ix],
            w.cs[c[ix]],
            table="sales_pipeline",
        )
    won = (r.random(n) < np.clip(win, 0.03, 0.9)) & (close <= 729)
    stage = np.where(
        close > 729,
        np.where(created + duration // 2 > 729, "Qualified", "Proposal"),
        np.where(won, "Closed Won", "Closed Lost"),
    )
    value = money(
        w.price[p]
        * np.maximum(1, r.negative_binomial(3, 0.35, n))
        * (1 + 0.6 * w.cs[c])
    )
    sales = w.employee(0, region, r)
    w.opp = dict(
        c=c,
        p=p,
        created=created,
        close=close,
        won=won,
        campaign=campaign,
        value=value,
        sales=sales,
        stage=stage,
    )
    w.opp_by_day = {d: np.flatnonzero(won & (close == d)) for d in range(730)}
    expected = created + np.clip(duration + r.normal(0, 10, n).astype(int), 1, 300)
    probability = np.where(
        stage == "Closed Won",
        1.0,
        np.where(stage == "Closed Lost", 0.0, np.where(stage == "Proposal", 0.65, 0.3)),
    )
    w.frames["sales_pipeline"] = pl.DataFrame(
        dict(
            opportunity_id=np.arange(1, n + 1),
            customer_id=c + 1,
            sales_rep_id=sales,
            product_id=p + 1,
            campaign_id=campaign + 1,
            created_at=times(0, n, r) + created.astype("timedelta64[D]"),
            expected_close_date=date(expected),
            actual_close_date=date(np.minimum(close, 729)),
            stage=stage,
            probability=probability,
            deal_value=value,
            lost_reason=np.where(
                stage == "Closed Lost",
                r.choice(
                    ["Price", "Competition", "Budget", "Timing", "Technical fit"],
                    n,
                    p=[0.25, 0.23, 0.25, 0.17, 0.1],
                ),
                None,
            ).tolist(),
            source=CHANNELS[a["channel"][campaign]],
            region=REGIONS[region],
        )
    ).with_columns(
        pl.when(pl.col("stage").is_in(["Qualified", "Proposal"]))
        .then(None)
        .otherwise(pl.col("actual_close_date"))
        .alias("actual_close_date")
    )
    # An ordered history makes progression investigable instead of just a terminal label.
    parts = []
    for j, (label, frac) in enumerate(
        [
            ("Qualified", 0.0),
            ("Discovery", 0.2),
            ("Proposal", 0.55),
            ("Negotiation", 0.8),
            ("Terminal", 1.0),
        ]
    ):
        d = created + (duration * frac).astype(int)
        mask = d <= 729
        ids = np.flatnonzero(mask)
        parts.append(
            pl.DataFrame(
                dict(
                    pipeline_event_id=ids * 5 + j + 1,
                    opportunity_id=ids + 1,
                    event_date=date(d[mask]),
                    stage=np.where(label == "Terminal", stage[mask], label),
                )
            )
        )
    w.frames["pipeline_stage_history"] = pl.concat(parts).sort("pipeline_event_id")

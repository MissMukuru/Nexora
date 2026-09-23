import numpy as np
import polars as pl
from .config import *


def initialize(w):
    r = w.r("marketing")
    n = w.cfg.count("campaigns")
    start = r.integers(0, 700, n)
    end = np.minimum(729, start + r.integers(7, 46, n))
    region = r.integers(0, 6, n)
    segment = r.choice(4, n, p=[0.2, 0.4, 0.28, 0.12])
    product = r.integers(0, w.cfg.count("products"), n)
    channel = r.choice(6, n, p=[0.25, 0.24, 0.22, 0.13, 0.1, 0.06])
    kind = r.choice(
        ["Acquisition", "Expansion", "Retention", "Launch"],
        n,
        p=[0.55, 0.22, 0.13, 0.1],
    )
    budget = money(r.lognormal(12.1, 0.8, n) * (1 + 0.45 * segment))
    overlap = np.maximum(0, np.minimum(end, 515) - np.maximum(start, 485) + 1) / (
        end - start + 1
    )
    history = np.ones(6)
    quality = np.empty(n)
    for i in np.argsort(start):
        quality[i] = np.clip(
            0.55 * history[channel[i]] + 0.45 * r.lognormal(0, 0.4), 0.25, 2.5
        )
        history[channel[i]] = 0.9 * history[channel[i]] + 0.1 * quality[i]
    quality *= np.array([1.1, 0.7, 1.05, 1.3, 0.98, 0.95])[channel] * (
        1 - 0.62 * overlap
    )
    quality *= np.array([1.08, 1.03, 0.95, 0.91, 1.01, 0.93])[region]
    quality *= (
        1
        + 0.14 * ((channel == 3) & (segment >= 2))
        + 0.12 * np.cos(2 * np.pi * start / 365 + w.pc[product])
    )
    quality *= np.clip(w.product_pop[product] ** 0.12, 0.7, 1.4)
    spend = money(budget * r.beta(8, 2, n) * (1 + 1.05 * overlap))
    # Episode factors applied by start date, before downstream funnel sampling.
    for day in np.unique(start):
        ix = np.flatnonzero(start == day)
        w.ep.set_day(int(day))
        f = w.ep.factor(
            "campaigns",
            int(day),
            region[ix],
            product=product[ix],
            segment=segment[ix],
            table="marketing_campaigns",
        )
        spend[ix] = money(spend[ix] * f)
        quality[ix] /= np.sqrt(f)
    w.campaign = dict(
        start=start,
        end=end,
        region=region,
        segment=segment,
        product=product,
        channel=channel,
        kind=kind,
        budget=budget,
        spend=spend,
        quality=quality,
        overlap=overlap,
    )
    w.campaign_revenue = np.zeros(n)
    w.campaign_conversions = np.zeros(n, dtype=int)
    w.ep.benchmark("EVT-003", "marketing_campaigns", int((overlap > 0).sum()))


def finalize(w):
    a = w.campaign
    n = len(a["start"])
    r = w.r("marketing-funnel")
    # Each pipeline opportunity represents a lead that passed qualification.
    qualified = np.bincount(w.opp["campaign"], minlength=n)
    rate = np.clip(0.28 * a["quality"] * (1 - 0.25 * a["overlap"]), 0.03, 0.68)
    leads = qualified + r.negative_binomial(np.maximum(qualified, 1), rate)
    clicks = leads + r.negative_binomial(np.maximum(leads, 1), 0.12)
    impressions = clicks + r.negative_binomial(np.maximum(clicks, 1), 0.025)
    conversions = w.campaign_conversions
    return pl.DataFrame(
        dict(
            campaign_id=np.arange(1, n + 1),
            campaign_name=[f"Nexora Campaign {i+1:05d}" for i in range(n)],
            channel=CHANNELS[a["channel"]],
            campaign_type=a["kind"],
            target_segment=SEGMENTS[a["segment"]],
            target_region=REGIONS[a["region"]],
            product_id=a["product"] + 1,
            start_date=date(a["start"]),
            end_date=date(a["end"]),
            budget=a["budget"],
            spend=a["spend"],
            impressions=impressions,
            clicks=clicks,
            leads=leads,
            qualified_leads=qualified,
            conversions=conversions,
            conversion_rate=conversions / np.maximum(leads, 1),
            customer_acquisition_cost=np.where(
                conversions > 0, a["spend"] / np.maximum(conversions, 1), np.nan
            ),
            revenue=money(w.campaign_revenue),
            roi=(w.campaign_revenue - a["spend"]) / np.maximum(a["spend"], 0.01),
        )
    ).with_columns(pl.col("customer_acquisition_cost").fill_nan(None))

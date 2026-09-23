import numpy as np
import polars as pl
from .config import *


def daily(w, day, n, writer):
    r = w.r("transactions", day)
    won = w.opp_by_day[day]
    if len(won) > n:
        raise ValueError("Daily transaction budget below won opportunities")
    c = w.customers_for(n, day, r)
    p = np.empty(n, dtype=int)
    # Industry/segment-specific category demand; product popularity has a heavy tail.
    cat_weights = np.array(
        [
            [0.30, 0.29, 0.07, 0.20, 0.14],
            [0.32, 0.29, 0.11, 0.12, 0.16],
            [0.32, 0.22, 0.20, 0.10, 0.16],
            [0.32, 0.17, 0.28, 0.06, 0.17],
        ]
    )
    cat = np.sum(r.random(n)[:, None] > np.cumsum(cat_weights[w.cs[c]], axis=1), axis=1)
    tech = (w.industry[c] == "Technology") & (r.random(n) < 0.25)
    cat[tech] = 0
    education = (w.industry[c] == "Education") & (r.random(n) < 0.30)
    cat[education] = 3
    month = int(date(day).astype("datetime64[M]").astype(int) % 12 + 1)
    if month in [1, 9]:
        cat[(w.industry[c] == "Education") & (r.random(n) < 0.2)] = 3
    for k in range(5):
        ix = np.flatnonzero(cat == k)
        pool = np.flatnonzero(w.pc == k)
        weights = w.product_pop[pool].copy()
        if k == 0 and day >= 455:
            # Relative elasticity favors cheaper software after the price change.
            weights *= np.exp(-0.12 * np.sqrt(w.price[pool] / np.median(w.price[pool])))
        p[ix] = r.choice(pool, len(ix), p=weights / weights.sum())
    repeat = (r.random(n) < 0.46) & (w.owned_count[c] > 0)
    p[repeat] = w.owned_product(c[repeat], r)
    # Win provenance is never fabricated by random ID assignment.
    k = len(won)
    c[:k] = w.opp["c"][won]
    p[:k] = w.opp["p"][won]
    regions = w.cr[c]
    seg = w.cs[c]
    ts = times(day, n, r)
    f = w.ep.factor("demand", day, regions, c, p, seg, ts, "transactions")
    # Demand shocks affect purchase quantities; external conditions also affect frequency
    # through the customer selection process.
    mean = (
        np.array([1.3, 2.3, 4.0, 7.0])[seg]
        * np.array([1.2, 1.0, 0.45, 0.8, 0.6])[w.pc[p]]
    )
    mean *= np.where(w.pc[p] == 0, 1 - 0.08 * (day >= 455) * (1 - seg * 0.12), 1)
    mean *= np.exp(0.009 * (w.ext["confidence"][day, regions] - 60)) * f
    q = 1 + r.negative_binomial(2, 2 / (2 + mean), n)
    inflation = w.ext["inflation"][day, regions]
    price = money(
        w.price[p]
        * np.exp(0.006 * (inflation - 5))
        * np.where((w.pc[p] == 0) & (day >= 455), 1.12, 1)
        * r.lognormal(0, 0.025, n)
    )
    channel = r.choice(6, n, p=[0.22, 0.12, 0.17, 0.24, 0.08, 0.17])
    channel[seg == 3] = np.where(
        r.random((seg == 3).sum()) < 0.55, 3, channel[seg == 3]
    )
    campaign = np.zeros(n, dtype="int64")
    opportunity = np.zeros(n, dtype="int64")
    campaign[:k] = w.opp["campaign"][won] + 1
    opportunity[:k] = won + 1
    channel[:k] = w.campaign["channel"][campaign[:k] - 1]
    discount = np.clip(
        r.beta(2, 20, n) + 0.025 * seg + 0.014 * np.log1p(q) + 0.025 * (channel == 3),
        0,
        0.6,
    )
    discount *= w.ep.factor("discount", day, regions, c, p, seg, ts, "transactions")
    discount = np.clip(discount, 0, 0.78)
    gross = money(q * price)
    reduction = money(gross * discount)
    net = money(gross - reduction)
    unit_cost = (
        w.cost[p]
        * (1 + 0.0015 * (w.ext["fuel"][day, regions] - 178) * (w.pc[p] == 1))
        * (1 + 0.002 * (w.ext["fx"][day, regions] - 129) * (w.pc[p] == 1))
    )
    cost = money(q * unit_cost * r.lognormal(0, 0.018, n))
    margin = money(net - cost)
    tid = w.next_ids("transactions", n)
    status = inventory_fulfill(w, day, p, regions, q, tid, writer)
    # Wins not fulfilled are retained as backorders, but excluded from recognized campaign revenue.
    fulfilled = status == "Completed"
    w.remember(c[fulfilled], p[fulfilled])
    np.add.at(w.revenue, c[fulfilled], net[fulfilled])
    w.last_purchase[c] = day
    sales = w.employee(0, regions, r)
    sales[:k] = w.opp["sales"][won]
    tracked = (campaign > 0) & fulfilled
    np.add.at(w.campaign_revenue, campaign[tracked] - 1, net[tracked])
    np.add.at(w.campaign_conversions, campaign[tracked] - 1, 1)
    if day >= 455:
        w.ep.benchmark("EVT-002", "transactions", int((w.pc[p] == 0).sum()))
    return pl.DataFrame(
        dict(
            transaction_id=tid,
            customer_id=c + 1,
            transaction_ts=ts,
            product_id=p + 1,
            quantity=q,
            unit_price=price,
            gross_amount=gross,
            discount_rate=discount,
            discount_amount=reduction,
            net_amount=net,
            cost_amount=cost,
            gross_margin=margin,
            payment_method=r.choice(
                ["M-Pesa", "Bank transfer", "Card", "Credit account"],
                n,
                p=[0.32, 0.30, 0.20, 0.18],
            ),
            sales_channel=CHANNELS[channel],
            sales_rep_id=sales,
            region=REGIONS[regions],
            currency=np.full(n, "KES"),
            transaction_status=status,
            campaign_id=campaign,
            opportunity_id=opportunity,
        )
    ).with_columns(
        [
            pl.when(pl.col(col) == 0).then(None).otherwise(pl.col(col)).alias(col)
            for col in ["campaign_id", "opportunity_id"]
        ]
    )


def inventory_fulfill(*args):
    from .inventory import fulfill

    return fulfill(*args)

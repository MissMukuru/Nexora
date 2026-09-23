import numpy as np
import polars as pl
from .config import *


def initialize(w):
    r = w.r("subscriptions")
    n = w.cfg.count("subscriptions")
    c = r.choice(w.cfg.count("customers"), n, p=w.propensity / w.propensity.sum())
    recurring = np.flatnonzero(w.recurring)
    p = r.choice(
        recurring, n, p=w.product_pop[recurring] / w.product_pop[recurring].sum()
    )
    start = np.maximum(
        np.where(r.random(n) < 0.34, 0, r.integers(0, 700, n)), np.maximum(0, w.acq[c])
    )
    plan = np.clip(w.cs[c] + r.integers(-1, 2, n), 0, 3)
    w.sub = dict(
        c=c,
        p=p,
        start=start,
        renew=start + 30,
        end=np.full(n, -1),
        active=np.zeros(n, dtype=bool),
        plan=plan,
        discount=r.beta(2, 18, n) * 0.4,
        auto=r.random(n) < 0.86,
        reason=np.full(n, "", dtype="<U30"),
        status=np.full(n, "Pending", dtype="<U12"),
        last_usage=np.full(n, -1),
        reactivated=np.zeros(n, dtype=bool),
    )
    w.sub["price"] = money(w.price[p] * (0.65 + 0.4 * plan))
    w.sub["initial_plan"] = plan.copy()
    w.sub["initial_price"] = w.sub["price"].copy()


def daily(w, day, writer):
    a = w.sub
    r = w.r("subscriptions", day)
    c = a["c"]
    p = a["p"]
    events = []

    def emit(ix, kind, old_plan=None, old_price=None):
        if not len(ix):
            return
        events.append(
            pl.DataFrame(
                dict(
                    subscription_event_id=w.next_ids("subscription_events", len(ix)),
                    subscription_id=ix + 1,
                    customer_id=c[ix] + 1,
                    product_id=p[ix] + 1,
                    event_date=np.repeat(date(day), len(ix)),
                    event_type=np.repeat(kind, len(ix)),
                    old_plan=a["plan"][ix] if old_plan is None else old_plan,
                    new_plan=a["plan"][ix],
                    old_monthly_price=(
                        a["price"][ix] if old_price is None else old_price
                    ),
                    new_monthly_price=a["price"][ix],
                )
            )
        )

    ix = np.flatnonzero(a["start"] == day)
    a["active"][ix] = True
    a["status"][ix] = "Active"
    if day >= 455:
        software = ix[w.pc[p[ix]] == 0]
        a["price"][software] = money(a["price"][software] * 1.12)
    w.remember(c[ix], p[ix])
    emit(ix, "Started")
    active = np.flatnonzero(a["active"])
    cc = c[active]
    if day == 455:
        ix = active[(w.pc[p[active]] == 0) & (a["start"][active] < day)]
        old = a["price"][ix].copy()
        a["price"][ix] = money(old * 1.12)
        emit(ix, "Price changed", old_price=old)
        w.ep.benchmark("EVT-002", "subscription_events", len(ix))
    eligible = active[a["renew"][active] <= day]
    cc = c[eligible]
    pp = p[eligible]
    idle = np.minimum(90, day - a["last_usage"][eligible]) / 90
    f = w.ep.factor(
        "subscriptions", day, w.cr[cc], cc, pp, w.cs[cc], table="subscriptions"
    )
    hazard = (
        sigmoid(
            -4.8
            + 1.9 * (1 - w.engagement[cc])
            + 0.45 * w.support_burden[cc]
            + 0.65 * w.payment_friction[cc]
            + 0.55 * (4 - w.satisfaction[cc])
            + 0.6 * idle
            + 0.28 * ((day >= 455) & (w.pc[pp] == 0))
            - 0.2 * w.cs[cc]
            - 0.001 * np.minimum(day - a["start"][eligible], 365)
        )
        * f
    )
    cancel = r.random(len(eligible)) < np.clip(hazard, 0.002, 0.65)
    expired = (~a["auto"][eligible]) & (r.random(len(eligible)) < 0.28)
    stop = eligible[cancel | expired]
    a["active"][stop] = False
    a["end"][stop] = day
    a["status"][stop] = np.where(expired[cancel | expired], "Expired", "Cancelled")
    a["reason"][stop] = np.where(
        w.payment_friction[c[stop]] > 0.7,
        "Payment friction",
        np.where(
            w.support_burden[c[stop]] > 1,
            "Support experience",
            np.where(
                w.engagement[c[stop]] < 0.4,
                "Low adoption",
                np.where(
                    (day >= 455) & (w.pc[p[stop]] == 0),
                    "Price/value",
                    "Business change",
                ),
            ),
        ),
    )
    emit(stop, "Ended")
    keep = eligible[~(cancel | expired)]
    a["renew"][keep] = day + 30
    emit(keep, "Renewed")
    u = r.random(len(keep))
    ups = (u < 0.035 + 0.035 * w.engagement[c[keep]]) & (a["plan"][keep] < 3)
    downs = (u > 0.972) & (a["plan"][keep] > 0)
    for mask, change, label in [(ups, 1, "Upgraded"), (downs, -1, "Downgraded")]:
        ix = keep[mask]
        old = a["plan"][ix].copy()
        oldprice = a["price"][ix].copy()
        a["plan"][ix] += change
        a["price"][ix] = money(
            a["price"][ix] * (0.65 + 0.4 * a["plan"][ix]) / (0.65 + 0.4 * old)
        )
        emit(ix, label, old, oldprice)
    ix = np.flatnonzero(
        (~a["active"]) & (a["end"] >= 0) & (day - a["end"] >= 45) & (~a["reactivated"])
    )
    ix = ix[r.random(len(ix)) < 0.0015]
    a["active"][ix] = True
    a["reactivated"][ix] = True
    a["status"][ix] = "Active"
    a["renew"][ix] = day + 30
    a["end"][ix] = -1
    a["reason"][ix] = ""
    emit(ix, "Reactivated")
    w.remember(c[ix], p[ix])
    # A customer is considered churned only if every started subscription is inactive.
    started = np.bincount(c[a["start"] <= day], minlength=len(w.cs))
    live = np.bincount(c[a["active"]], minlength=len(w.cs))
    w.churned = (started > 0) & (live == 0)
    if events:
        writer.write("subscription_events", pl.concat(events), day)


def finalize(w):
    a = w.sub
    end = a["end"]
    n = len(end)
    return pl.DataFrame(
        dict(
            subscription_id=np.arange(1, n + 1),
            customer_id=a["c"] + 1,
            product_id=a["p"] + 1,
            plan=np.array(["Basic", "Standard", "Pro", "Enterprise"])[a["plan"]],
            start_date=date(a["start"]),
            renewal_date=date(a["renew"]),
            end_date=date(np.where(end >= 0, end, 0)),
            status=a["status"],
            monthly_price=a["price"],
            discount_rate=a["discount"],
            cancellation_reason=np.where(end >= 0, a["reason"], None).tolist(),
            auto_renew=a["auto"],
            tenure_days=np.where(end >= 0, end, 729) - a["start"],
        )
    ).with_columns(
        pl.when(pl.col("status") == "Active")
        .then(None)
        .otherwise(pl.col("end_date"))
        .alias("end_date")
    )

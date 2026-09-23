import numpy as np
import polars as pl
from .config import *


def daily(w, day, n):
    r = w.r("usage", day)
    a = w.sub
    active = np.flatnonzero(a["active"])
    if not len(active):
        raise ValueError("Usage requires active subscriptions")
    weights = (0.2 + w.engagement[a["c"][active]]) * (1 + 0.4 * a["plan"][active])
    # Session grain, multiple sessions on a date are legitimate, not duplicate source rows.
    s = r.choice(active, n, p=weights / weights.sum())
    c = a["c"][s]
    p = a["p"][s]
    regions = w.cr[c]
    factor = w.ep.factor(
        "usage", day, regions, c, p, w.cs[c], table="customer_product_usage"
    )
    mu = np.maximum(
        0.1,
        (5 + 3 * a["plan"][s])
        * w.engagement[c]
        * factor
        / (1 + 0.14 * w.support_burden[c]),
    )
    events = r.negative_binomial(3, 3 / (3 + mu), n)
    users = np.minimum(w.size[c], 1 + r.poisson(1 + a["plan"][s], n))
    users = np.where(events == 0, 0, np.minimum(users, events))
    minutes = np.round(events * r.gamma(2, 4, n), 2)
    adoption = np.clip(
        r.beta(3, 3, n) * (0.45 + w.engagement[c]) + 0.04 * a["plan"][s], 0, 1
    )
    error = np.clip(
        r.beta(1, 55, n)
        * (1 + 0.35 * w.pressure[regions] + w.support_burden[c])
        / np.sqrt(factor),
        0,
        1,
    )
    a["last_usage"][s[events > 0]] = day
    return pl.DataFrame(
        dict(
            usage_id=w.next_ids("customer_product_usage", n),
            subscription_id=s + 1,
            customer_id=c + 1,
            product_id=p + 1,
            date=np.repeat(date(day), n),
            usage_events=events,
            active_users=users,
            usage_minutes=minutes,
            feature_adoption_rate=adoption,
            error_rate=error,
        )
    )

import numpy as np
import polars as pl
from .config import *


def advance(w, day):
    for c, satisfaction in w.support_updates.pop(day, []):
        count = np.bincount(c, minlength=len(w.cs))
        scores = np.bincount(c, weights=satisfaction, minlength=len(w.cs))
        affected = count > 0
        w.satisfaction[affected] = (
            0.8 * w.satisfaction[affected] + 0.2 * scores[affected] / count[affected]
        )


def daily(w, day, n, tx):
    r = w.r("support", day)
    # Cases are sampled from actual purchases today; repeated case-to-order links
    # model distinct incidents and preserve customer-product ownership evidence.
    tc = tx["customer_id"].to_numpy() - 1
    pp = tx["product_id"].to_numpy() - 1
    regions = w.cr[tc]
    pressure = w.pressure[regions]
    weights = (
        (0.25 + w.support_burden[tc])
        * (1 + 0.4 * pressure)
        * (1 + 0.5 * (tx["transaction_status"].to_numpy() == "Backordered"))
    )
    weights *= w.ep.factor(
        "support",
        day,
        regions,
        tc,
        pp,
        w.cs[tc],
        tx["transaction_ts"].to_numpy(),
        "support_cases",
    )
    ix = r.choice(len(tc), n, p=weights / weights.sum())
    c = tc[ix]
    p = pp[ix]
    reg = w.cr[c]
    severity = np.clip(r.poisson(0.6 + 0.1 * w.pressure[reg], n) + 1, 1, 4)
    issue = r.choice(
        ["Access", "Billing", "Performance", "Configuration", "Delivery", "Defect"],
        n,
        p=[0.16, 0.17, 0.23, 0.19, 0.13, 0.12],
    )
    issue = np.where(
        w.pc[p] == 1, np.where(r.random(n) < 0.6, "Delivery", "Defect"), issue
    )
    opened = tx["transaction_ts"].to_numpy()[ix].astype("datetime64[s]") + (
        r.integers(0, 1800, n)
    ).astype("timedelta64[s]")
    opened = np.minimum(
        opened, date(day).astype("datetime64[s]") + np.timedelta64(86399, "s")
    )
    factor = w.ep.factor(
        "support", day, reg, c, p, w.cs[c], opened, "support_resolution"
    )
    duration = np.maximum(
        0.1,
        r.lognormal(1.4 + 0.22 * w.pressure[reg] + 0.18 * severity, 0.65, n) * factor,
    )
    if 438 <= day <= 469:
        duration[reg == 0] *= 1.65
        w.ep.benchmark("EVT-001", "support_cases", int((reg == 0).sum()))
    close = opened + (duration * 3600).astype("timedelta64[s]")
    resolved = close < (END + 1).astype("datetime64[s]")
    satisfaction = np.clip(
        np.rint(5 - 0.032 * duration - 0.12 * severity + r.normal(0, 0.65, n)), 1, 5
    )
    escalated = (duration > 48) | (severity == 4) | (r.random(n) < 0.03)
    # Update next-day customer state from observable current support pressure.
    count = np.bincount(c, minlength=len(w.cs))
    scores = np.bincount(c, weights=satisfaction, minlength=len(w.cs))
    affected = count > 0
    w.support_burden[affected] = np.clip(
        w.support_burden[affected] + 0.15 * np.log1p(count[affected]), 0, 5
    )
    close_day = (close.astype("datetime64[D]") - START).astype(int)
    for d in np.unique(close_day[resolved]):
        closedmask = resolved & (close_day == d)
        w.support_updates.setdefault(max(day + 1, int(d)), []).append(
            (c[closedmask], satisfaction[closedmask])
        )
    frame = pl.DataFrame(
        dict(
            case_id=w.next_ids("support_cases", n),
            customer_id=c + 1,
            product_id=p + 1,
            transaction_id=tx["transaction_id"].to_numpy()[ix],
            employee_id=w.employee(1, reg, r),
            opened_at=opened.astype("datetime64[ms]"),
            closed_at=close.astype("datetime64[ms]"),
            issue_category=issue,
            severity=severity,
            status=np.where(resolved, "Resolved", "Open"),
            escalation_flag=escalated,
            resolution_time=np.round(duration, 3),
            satisfaction_score=satisfaction,
        )
    ).with_columns(
        [
            pl.when(pl.col("status") == "Open")
            .then(None)
            .otherwise(pl.col(col))
            .alias(col)
            for col in ["closed_at", "resolution_time", "satisfaction_score"]
        ]
    )
    return frame

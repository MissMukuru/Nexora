import numpy as np
import polars as pl
from .config import *


def daily(w, day, n, cases, tx):
    r = w.r("interactions", day)
    linked = min(n, len(cases))
    extra = n - linked
    # Every case has a real interaction; additional contacts include sales/billing/feedback.
    c = np.r_[
        cases["customer_id"].to_numpy() - 1,
        tx["customer_id"].to_numpy()[r.integers(0, len(tx), extra)] - 1,
    ][:n]
    reg = w.cr[c]
    case_id = np.r_[cases["case_id"].to_numpy(), np.zeros(extra, dtype="int64")][:n]
    employee = np.r_[cases["employee_id"].to_numpy(), w.employee(1, reg[linked:], r)][
        :n
    ]
    kind = np.r_[
        np.full(linked, "Support"),
        r.choice(
            ["Sales", "Billing", "Feedback", "Onboarding"],
            extra,
            p=[0.32, 0.24, 0.18, 0.26],
        ),
    ]
    issue = np.r_[
        cases["issue_category"].to_numpy(),
        r.choice(["Inquiry", "Invoice", "Feedback", "Setup"], extra),
    ][:n]
    ts = times(day, n, r)
    ts[:linked] = cases["opened_at"].to_numpy()[:linked].astype("datetime64[s]")
    response = r.lognormal(0.1 + 0.38 * w.pressure[reg], 0.65, n)
    duration = np.r_[
        cases["resolution_time"].fill_null(float("nan")).to_numpy(),
        r.lognormal(1, 0.6, extra),
    ][:n]
    status = np.r_[cases["status"].to_numpy(), np.full(extra, "Resolved")][:n]
    sat = np.r_[
        cases["satisfaction_score"].fill_null(float("nan")).to_numpy(),
        np.clip(np.rint(w.satisfaction[c[linked:]] + r.normal(0, 0.6, extra)), 1, 5),
    ][:n]
    sentiment = np.clip(
        (np.nan_to_num(sat, nan=2) - 3) / 2
        + r.normal(0, 0.2, n)
        - 0.05 * w.pressure[reg],
        -1,
        1,
    )
    response = np.minimum(response, np.nan_to_num(duration, nan=response))
    escalated = np.r_[cases["escalation_flag"].to_numpy(), r.random(extra) < 0.03][:n]
    priority = np.where(escalated, "High", np.where(w.cs[c] >= 2, "Medium", "Low"))
    if 438 <= day <= 469:
        w.ep.benchmark("EVT-001", "interactions", int((reg == 0).sum()))
    return pl.DataFrame(
        dict(
            interaction_id=w.next_ids("interactions", n),
            customer_id=c + 1,
            employee_id=employee,
            case_id=case_id,
            interaction_ts=ts,
            interaction_type=kind,
            channel=r.choice(
                ["Phone", "Email", "Chat", "WhatsApp", "In person"],
                n,
                p=[0.18, 0.3, 0.22, 0.25, 0.05],
            ),
            priority=priority,
            issue_category=issue,
            status=status,
            sentiment_score=sentiment,
            first_response_time=np.round(response, 3),
            resolution_time=np.round(duration, 3),
            escalation_flag=escalated,
            satisfaction_score=sat,
            region=REGIONS[reg],
        )
    ).with_columns(
        pl.col("resolution_time", "satisfaction_score").fill_nan(None),
        pl.when(pl.col("case_id") == 0)
        .then(None)
        .otherwise(pl.col("case_id"))
        .alias("case_id"),
    )

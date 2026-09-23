import numpy as np
import polars as pl
from .config import *


def initialize(w):
    w.payment_updates = {}


def advance(w, day):
    for customers, delta in w.payment_updates.pop(day, []):
        np.add.at(w.payment_friction, customers, delta)
    w.payment_friction = np.clip(w.payment_friction, 0, 3)


def daily(w, day, tx):
    r = w.r("payments", day)
    n = len(tx)
    c = tx["customer_id"].to_numpy() - 1
    p = tx["product_id"].to_numpy() - 1
    reg = w.cr[c]
    terms = np.array([7, 14, 30, 45])[w.cs[c]]
    due = day + terms
    risk = (
        w.credit[c]
        + 0.06 * w.support_burden[c]
        + 0.02 * (w.ext["inflation"][day, reg] - 5)
    )
    factor = w.ep.factor("payments", day, reg, c, p, w.cs[c], table="invoices_payments")
    delay = np.maximum(
        0,
        np.rint(
            r.gamma(2, 5 + 5 * w.cs[c], n) * (1 + risk) * factor + r.normal(-5, 3, n)
        ),
    ).astype(int)
    intended = day + delay
    default = r.random(n) < np.clip(risk * 0.055, 0, 0.2)
    intended = np.where(default, 10000, intended)
    valid = tx["transaction_status"].to_numpy() == "Completed"
    paid = (intended <= 729) & valid
    partial = (~paid) & valid & (r.random(n) < 0.18) & (day + 4 <= 729)
    amount = tx["net_amount"].to_numpy()
    fraction = r.beta(3, 3, n)
    paid_amount = money(np.where(paid, amount, np.where(partial, amount * fraction, 0)))
    status = np.where(
        ~valid,
        "Void",
        np.where(
            paid,
            "Paid",
            np.where(partial, "Partially paid", np.where(due < 729, "Overdue", "Open")),
        ),
    )
    payment = np.where(paid, intended, np.where(partial, day + 4, -1))
    # Expose risk only after a due date passes, not from foreknowledge of final payment.
    overdue = (intended > due) & valid
    for d in np.unique(due[overdue] + 1):
        ix = overdue & (due + 1 == d)
        w.payment_updates.setdefault(int(d), []).append(
            (c[ix], np.full(ix.sum(), 0.05))
        )
    for d in np.unique(intended[overdue & paid]):
        ix = overdue & paid & (intended == d)
        w.payment_updates.setdefault(int(d), []).append(
            (c[ix], np.full(ix.sum(), -0.04))
        )
    return pl.DataFrame(
        dict(
            invoice_id=w.next_ids("invoices_payments", n),
            customer_id=c + 1,
            transaction_id=tx["transaction_id"].to_numpy(),
            invoice_date=np.repeat(date(day), n),
            due_date=date(due),
            payment_date=date(np.maximum(payment, 0)),
            invoice_amount=amount,
            paid_amount=paid_amount,
            payment_status=status,
            days_to_payment=np.where(payment >= 0, payment - day, -1),
            overdue_days=np.where(
                valid, np.maximum(0, np.where(paid, intended, 729) - due), 0
            ),
        )
    ).with_columns(
        pl.when(pl.col("days_to_payment") < 0)
        .then(None)
        .otherwise(pl.col("payment_date"))
        .alias("payment_date"),
        pl.when(pl.col("days_to_payment") < 0)
        .then(None)
        .otherwise(pl.col("days_to_payment"))
        .alias("days_to_payment"),
    )

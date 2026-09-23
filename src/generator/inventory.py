import numpy as np
import polars as pl
from .config import *


def initialize(w):
    r = w.r("inventory")
    w.hardware = np.flatnonzero(w.pc == 1)
    w.stock = np.zeros((w.cfg.count("products"), 6), dtype="int64")
    w.stock[w.hardware, :] = r.integers(350, 850, (len(w.hardware), 6))
    w.opening_stock = w.stock.copy()
    w.pending_receipts = {}
    w.reorder_pending = np.zeros_like(w.stock, dtype=bool)
    w.suppliers = w.r("suppliers").integers(1, 81, w.cfg.count("products"))
    w.frames["suppliers"] = pl.DataFrame(
        dict(
            supplier_id=np.arange(1, 81),
            supplier_name=[f"Supplier {i:03d}" for i in range(1, 81)],
            supplier_region=np.resize(REGIONS, 80),
            base_lead_time_days=np.resize([3, 5, 7, 10, 14], 80),
        )
    )


def frame(w, day, p, region, qty, kind, after, lead=None, transaction=None):
    n = len(p)
    return pl.DataFrame(
        dict(
            movement_id=w.next_ids("inventory_movements", n),
            product_id=p + 1,
            region=REGIONS[region],
            date=np.repeat(date(day), n),
            movement_type=np.broadcast_to(kind, (n,)),
            quantity=qty,
            unit_cost=money(
                w.cost[p] * (1 + 0.001 * (w.ext["fuel"][day, region] - 178))
            ),
            supplier_id=w.suppliers[p],
            lead_time_days=np.zeros(n, dtype=int) if lead is None else lead,
            stock_level_after=after,
            transaction_id=(
                np.zeros(n, dtype="int64") if transaction is None else transaction
            ),
        )
    ).with_columns(
        pl.when(pl.col("transaction_id") == 0)
        .then(None)
        .otherwise(pl.col("transaction_id"))
        .alias("transaction_id")
    )


def begin_day(w, day, writer):
    r = w.r("inventory", day)
    p = np.repeat(w.hardware, 6)
    region = np.tile(np.arange(6), len(w.hardware))
    if day == 0:
        writer.write(
            "inventory_movements",
            frame(w, day, p, region, w.stock[p, region], "Opening", w.stock[p, region]),
            day,
        )
    due = w.pending_receipts.pop(day, [])
    for pp, rr, qty, lead in due:
        w.stock[pp, rr] += qty
        w.reorder_pending[pp, rr] = False
        writer.write(
            "inventory_movements",
            frame(w, day, pp, rr, qty, "Receipt", w.stock[pp, rr], lead),
            day,
        )
    # Paired warehouse-bin rebalancing entries. Regional balance is unchanged;
    # order is explicit through movement_id, so every ledger prefix is valid.
    for phase in ["Reserve to pick", "Pick to reserve"]:
        qty = np.minimum(w.stock[p, region], r.poisson(3, len(p)) + 1)
        w.stock[p, region] -= qty
        writer.write(
            "inventory_movements",
            frame(w, day, p, region, -qty, phase + " out", w.stock[p, region]),
            day,
        )
        w.stock[p, region] += qty
        writer.write(
            "inventory_movements",
            frame(w, day, p, region, qty, phase + " in", w.stock[p, region]),
            day,
        )
    # Lead-time shocks propagate into stock availability and subsequent orders.
    factor = w.ep.factor(
        "inventory", day, region, product=p, table="inventory_movements"
    )
    reorder = (w.stock[p, region] < 260) & (~w.reorder_pending[p, region])
    pp = p[reorder]
    rr = region[reorder]
    if len(pp):
        lead = np.clip(
            np.rint(
                r.gamma(2.5, 2, len(pp)) * factor[reorder]
                + w.ext["traffic"][day, rr] * 2
            ),
            1,
            50,
        ).astype(int)
        qty = np.maximum(
            100, np.rint((650 - w.stock[pp, rr]) * r.lognormal(0, 0.2, len(pp)))
        ).astype(int)
        w.reorder_pending[pp, rr] = True
        for due_day in np.unique(day + lead):
            ix = day + lead == due_day
            w.pending_receipts.setdefault(int(due_day), []).append(
                (pp[ix], rr[ix], qty[ix], lead[ix])
            )


def fulfill(w, day, p, regions, q, tid, writer):
    status = np.full(len(p), "Completed", dtype="<U20")
    hw = np.flatnonzero(w.pc[p] == 1)
    if not len(hw):
        return status
    # Vectorized grouped cumulative allocation, sorted by product/region and original order.
    key = p[hw] * 6 + regions[hw]
    order = np.argsort(key, kind="stable")
    ix = hw[order]
    k = key[order]
    boundaries = np.r_[0, np.flatnonzero(k[1:] != k[:-1]) + 1]
    cumulative = np.cumsum(q[ix])
    starts = np.repeat(
        np.r_[0, cumulative[boundaries[1:] - 1]], np.diff(np.r_[boundaries, len(ix)])
    )
    demand = cumulative - starts
    available = w.stock[p[ix], regions[ix]]
    accepted = demand <= available
    # Once a group exhausts its stock, remaining orders are backordered, no negative stock.
    status[ix[~accepted]] = "Backordered"
    shipped = ix[accepted]
    after = available[accepted] - demand[accepted]
    if len(shipped):
        writer.write(
            "inventory_movements",
            frame(
                w,
                day,
                p[shipped],
                regions[shipped],
                -q[shipped],
                "Sale",
                after,
                transaction=tid[shipped],
            ),
            day,
        )
        np.add.at(w.stock, (p[shipped], regions[shipped]), -q[shipped])
    return status

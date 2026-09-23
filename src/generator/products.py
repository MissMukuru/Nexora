import numpy as np
import polars as pl
from .config import *


def generate(w):
    n = w.cfg.count("products")
    r = w.r("products")
    ids = np.arange(1, n + 1)
    cat = np.arange(n) % 5
    price = r.lognormal(np.array([8.6, 10.1, 11.2, 9.1, 8.8])[cat], 0.65)
    margin = np.clip(
        r.normal(np.array([0.78, 0.26, 0.48, 0.6, 0.57])[cat], 0.07), 0.12, 0.92
    )
    w.pc = cat
    w.price = money(price)
    w.cost = money(price * (1 - margin))
    w.recurring = np.isin(cat, [0, 4])
    w.product_pop = r.pareto(2.5, n) + 0.35
    w.launch = np.full(n, -1, dtype=int)
    w.frames["products_services"] = pl.DataFrame(
        dict(
            product_id=ids,
            product_name=[f"{CATEGORIES[c]} {i:04d}" for i, c in zip(ids, cat)],
            category=CATEGORIES[cat],
            business_unit=np.array(
                [
                    "Digital",
                    "Devices",
                    "Professional Services",
                    "Learning",
                    "Customer Success",
                ]
            )[cat],
            product_family=[
                f"{CATEGORIES[c]} Family {i%13+1}" for i, c in zip(ids, cat)
            ],
            cost=w.cost,
            base_price=w.price,
            launch_date=date(-r.integers(1, 1800, n)),
            lifecycle_stage=r.choice(
                ["Growth", "Mature", "Legacy"], n, p=[0.3, 0.58, 0.12]
            ),
            margin_target=margin,
            recurring_flag=w.recurring,
        )
    )

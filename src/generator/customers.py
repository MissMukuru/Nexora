import numpy as np
import polars as pl
from .config import *


def generate(w):
    n = w.cfg.count("customers")
    r = w.r("customers")
    ids = np.arange(1, n + 1)
    region = r.choice(6, n, p=[0.38, 0.19, 0.12, 0.09, 0.13, 0.09])
    size = r.lognormal(3.4 + 0.32 * (region == 0), 1.35, n).astype(int) + 2
    segment = np.digitize(size, [20, 100, 500])
    industry = r.choice(
        [
            "Finance",
            "Retail",
            "Healthcare",
            "Education",
            "Manufacturing",
            "Technology",
            "Logistics",
            "Hospitality",
        ],
        n,
        p=[0.12, 0.20, 0.12, 0.13, 0.12, 0.13, 0.1, 0.08],
    )
    size = np.where(industry == "Manufacturing", (size * 1.6).astype(int), size)
    segment = np.digitize(size, [20, 100, 500])
    acq = np.where(r.random(n) < 0.78, -r.integers(1, 2200, n), r.integers(0, 700, n))
    # Guarantee targetable founding cohorts, including every regional segment.
    for k in range(min(24, n)):
        region[k] = k // 4
        segment[k] = k % 4
        size[k] = [8, 45, 240, 1200][k % 4]
        acq[k] = -500
    channel = r.choice(6, n, p=[0.24, 0.17, 0.13, 0.23, 0.1, 0.13])
    channel = np.where((segment == 3) & (r.random(n) < 0.5), 3, channel)
    w.cr = region
    w.cs = segment
    w.acq = acq
    w.size = size
    w.industry = industry
    w.propensity = (
        r.lognormal(-0.2, 0.75, n)
        * (1 + 0.8 * segment)
        * (1 + 0.18 * (industry == "Technology"))
        * (1 + 0.15 * (channel == 3))
    )
    w.frames["customers"] = pl.DataFrame(
        dict(
            customer_id=ids,
            company_name=[f"Nexora Client {x:06d}" for x in ids],
            segment=SEGMENTS[segment],
            region=REGIONS[region],
            industry=industry,
            company_size=np.array(["Micro", "Small", "Medium", "Large"])[segment],
            employee_count=size,
            annual_revenue_band=np.array(
                ["<10M KES", "10M-100M KES", "100M-1B KES", ">1B KES"]
            )[segment],
            acquisition_date=date(acq),
            acquisition_channel=CHANNELS[channel],
            customer_lifetime_value=np.zeros(n),
            account_status=np.full(n, "Active"),
        )
    )

import numpy as np
import polars as pl
from .config import *


def generate(w):
    n = w.cfg.count("employees")
    r = w.r("employees")
    ids = np.arange(1, n + 1)
    dep = r.choice(6, n, p=[0.22, 0.3, 0.24, 0.09, 0.09, 0.06])
    reg = r.choice(6, n, p=[0.4, 0.17, 0.12, 0.08, 0.14, 0.09])
    for i in range(min(36, n)):
        dep[i] = i // 6
        reg[i] = i % 6
    level = r.choice(5, n, p=[0.23, 0.4, 0.23, 0.11, 0.03])
    tenure = r.gamma(2, 25, n).astype(int) + 24
    w.ed = dep
    w.er = reg
    w.frames["employees"] = pl.DataFrame(
        dict(
            employee_id=ids,
            employee_name=[f"Employee {i:04d}" for i in ids],
            department=np.array(
                [
                    "Sales",
                    "Support",
                    "Operations",
                    "Engineering",
                    "Marketing",
                    "Finance",
                ]
            )[dep],
            job_level=np.array(
                ["Associate", "Specialist", "Senior", "Manager", "Director"]
            )[level],
            region=REGIONS[reg],
            contract_type=r.choice(
                ["Permanent", "Fixed term", "Contractor"], n, p=[0.74, 0.17, 0.09]
            ),
            hire_date=END - tenure.astype("timedelta64[M]").astype("timedelta64[D]"),
            annual_salary=money(
                r.lognormal(
                    np.log(
                        np.array([900000, 720000, 600000, 1700000, 950000, 1100000])[
                            dep
                        ]
                        * (1 + 0.6 * level)
                    ),
                    0.25,
                )
            ),
            performance_band=r.choice(
                ["Developing", "Meets", "Exceeds", "Exceptional"],
                n,
                p=[0.12, 0.55, 0.28, 0.05],
            ),
            tenure_months=tenure,
        )
    )

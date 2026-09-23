"""Run with python -m src.generator.generate. Full production scale is the default."""

import argparse
import json
import os
from pathlib import Path
import shutil
import time
import platform
import numpy as np
import polars as pl
from .config import Config, DAYS, date, money
from .entities import World
from .anomalies import Episodes
from . import (
    customers,
    products,
    employees,
    external,
    marketing,
    pipeline,
    subscriptions,
    operations,
    inventory,
    transactions,
    usage,
    support,
    interactions,
    payments,
    ground_truth,
)
from .io import Writer


def run(cfg):
    started = time.perf_counter()
    output = cfg.root.resolve()
    staged = output.with_name(output.name + ".building")
    if output.exists() or staged.exists():
        raise FileExistsError(
            f"Refusing to overwrite {output} or {staged}. Choose a fresh --output."
        )
    if not 0.001 <= cfg.scale <= 1:
        raise ValueError("scale must be in [.001,1]; default 1 is production")
    if cfg.count("employees") < 36:
        raise ValueError(
            "Scale requires at least 36 employees; use --scale 0.02 or larger"
        )
    w = World(cfg)
    customers.generate(w)
    products.generate(w)
    employees.generate(w)
    w.init_state()
    w.ep = Episodes(w)
    external.generate(w)
    marketing.initialize(w)
    pipeline.initialize(w)
    subscriptions.initialize(w)
    operations.initialize(w)
    inventory.initialize(w)
    payments.initialize(w)
    writer = Writer(w, staged)
    simtimes = {}

    def timed(name, fn, *args):
        t = time.perf_counter()
        result = fn(*args)
        simtimes[name] = simtimes.get(name, 0) + time.perf_counter() - t
        return result

    try:
        for name in [
            "products_services",
            "employees",
            "suppliers",
            "external_factors",
            "sales_pipeline",
            "pipeline_stage_history",
        ]:
            writer.write(name, w.frames[name])
        # Release large pipeline/history frames after writing; compact arrays remain.
        del w.frames["pipeline_stage_history"]
        del w.frames["sales_pipeline"]
        txcounts = external.schedule(w, "transactions", cfg.count("transactions"))
        usagecounts = external.schedule(w, "usage", cfg.count("usage"))
        # Activity-driven support calendar; regional allocation later uses actual operational pressure.
        r = w.r("support-calendar")
        supportweights = np.power(txcounts, 0.82) * (
            1 + 0.10 * np.sin(np.arange(DAYS) / 35)
        )
        supportcounts = r.multinomial(
            cfg.count("support"), supportweights / supportweights.sum()
        )
        extras = external.schedule(
            w, "extra-interactions", cfg.count("interactions") - cfg.count("support")
        )
        for day in range(DAYS):
            w.ep.set_day(day)
            payments.advance(w, day)
            support.advance(w, day)
            r = w.r("customer-evolution", day)
            # Slow, noisy engagement with feedback and regression to a heterogeneous baseline.
            w.support_burden *= 0.96
            w.payment_friction *= 0.998
            target = (
                0.72
                - 0.035 * w.support_burden
                - 0.03 * w.payment_friction
                + 0.018 * (w.satisfaction - 4)
            )
            w.engagement = np.clip(
                0.994 * w.engagement + 0.006 * target + r.normal(0, 0.003, len(w.cs)),
                0.03,
                0.99,
            )
            timed("inventory", inventory.begin_day, w, day, writer)
            timed("subscriptions", subscriptions.daily, w, day, writer)
            tx = timed(
                "transactions", transactions.daily, w, day, int(txcounts[day]), writer
            )
            writer.write("transactions", tx, day)
            reg = np.array(
                [
                    np.count_nonzero(tx["region"].to_numpy() == name)
                    for name in [
                        "Nairobi",
                        "Kiambu",
                        "Nakuru",
                        "Eldoret",
                        "Mombasa",
                        "Kisumu",
                    ]
                ]
            )
            ops = timed("operations", operations.daily, w, day, reg)
            writer.write("daily_operations", ops, day)
            use = timed("usage", usage.daily, w, day, int(usagecounts[day]))
            writer.write("customer_product_usage", use, day)
            # Observable usage errors influence subsequent support and satisfaction.
            cc = use["customer_id"].to_numpy() - 1
            err = use["error_rate"].to_numpy()
            cnt = np.bincount(cc, minlength=len(w.cs))
            errors = np.bincount(cc, weights=err, minlength=len(w.cs))
            has = cnt > 0
            w.support_burden[has] = np.clip(
                w.support_burden[has] + errors[has] / cnt[has] * 0.12, 0, 5
            )
            observed = np.bincount(
                cc, weights=use["usage_events"].to_numpy(), minlength=len(w.cs)
            ) / np.maximum(cnt, 1)
            w.engagement[has] = np.clip(
                0.997 * w.engagement[has] + 0.003 * np.minimum(1, observed[has] / 8),
                0.03,
                0.99,
            )
            cases = timed("support", support.daily, w, day, int(supportcounts[day]), tx)
            writer.write("support_cases", cases, day)
            contacts = timed(
                "interactions",
                interactions.daily,
                w,
                day,
                len(cases) + int(extras[day]),
                cases,
                tx,
            )
            writer.write("interactions", contacts, day)
            invoice = timed("payments", payments.daily, w, day, tx)
            writer.write("invoices_payments", invoice, day)
            if day % 30 == 0 or day == 729:
                print(
                    f'{date(day)} | transactions={w.ids["transactions"]:,} | elapsed={time.perf_counter()-started:.1f}s',
                    flush=True,
                )
        w.frames["customers"] = w.frames["customers"].with_columns(
            pl.Series("customer_lifetime_value", money(w.revenue)),
            pl.Series(
                "account_status",
                np.where(
                    w.churned,
                    "Churned",
                    np.where(w.payment_friction > 1, "At risk", "Active"),
                ),
            ),
        )
        writer.write("customers", w.frames["customers"])
        writer.write("subscriptions", subscriptions.finalize(w))
        writer.write("marketing_campaigns", marketing.finalize(w))
        print("Running complete cross-table validation and profiles...", flush=True)
        report = writer.report()
        assert len(w.ep.items) == cfg.episodes
        assert all(
            sum(w.ep.counts[e["event_id"]].values()) > 0 for e in w.ep.benchmarks
        ), "Benchmark without exposure"
        report["episode_coverage"] = {
            "configured": len(w.ep.items),
            "exposed": sum(
                bool(sum(w.ep.counts[e["episode_id"]].values())) for e in w.ep.items
            ),
            "placebo": sum(e["placebo"] for e in w.ep.items),
        }
        report.update(
            config=cfg.metadata(),
            seed=cfg.seed,
            window=["2025-01-01", "2026-12-31"],
            simulation_time_seconds=simtimes,
            total_elapsed_seconds=time.perf_counter() - started,
            python=platform.python_version(),
            numpy=np.__version__,
            polars=pl.__version__,
            status="validated",
            row_count=sum(v["row_count"] for v in report["datasets"].values()),
        )
        ground_truth.write(w, staged)
        (staged / "generation_report.json").write_text(
            json.dumps(report, indent=2, default=str, allow_nan=False)
        )
        # Report itself is evaluation/administrative data and must not be ingested by agents.
        os.chmod(staged / "generation_report.json", 0o600)
        shutil.rmtree(staged / ".validation_tmp", ignore_errors=True)
        (staged / "_SUCCESS").write_text(
            "Validated canonical CSV. Ingest only raw/.\n"
        )
        os.rename(staged, output)
        print(
            f'COMPLETE: {report["row_count"]:,} canonical rows in {output}', flush=True
        )
        return report
    except BaseException:
        writer.close()
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="data")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="Explicit developer fixture only; production default is 1.0",
    )
    args = parser.parse_args()
    run(Config(output=args.output, seed=args.seed, scale=args.scale))


if __name__ == "__main__":
    main()

from pathlib import Path
import time
import json
import os
import numpy as np
import polars as pl
import duckdb
from .config import date
from .validation import Validator, PK, global_checks


class Writer:
    def __init__(self, w, root):
        self.w = w
        self.root = Path(root)
        self.validator = Validator(w)
        self.open = {}
        self.schema = {}
        self.elapsed = {}
        self.source_stats = {}
        self.root.mkdir(parents=True)

    def write(self, name, f, day=None):
        if not len(f):
            return
        tick = time.perf_counter()
        self.validator.validate(name, f)
        # Nullable telemetry only. Keys/arithmetic and business state are never corrupted.
        if name in ["interactions", "customer_product_usage"]:
            r = self.w.r("canonical-missingness", name, day)
            col = (
                "sentiment_score" if name == "interactions" else "feature_adoption_rate"
            )
            mask = r.random(len(f)) < 0.002
            f = f.with_columns(
                pl.when(pl.Series(mask)).then(None).otherwise(pl.col(col)).alias(col)
            )
        if day is not None:
            r = self.w.r("arrival", name, day)
            delays = np.where(r.random(len(f)) < 0.012, r.integers(2, 22, len(f)), 0)
            f = f.with_columns(
                pl.Series("ingested_at", date(day + delays).astype("datetime64[ms]"))
            )
        # Enforce a table's schema across null-heavy and differently-sized batches.
        if name in self.schema:
            f = f.cast(self.schema[name])
        else:
            self.schema[name] = f.schema
        dest = self.root / "raw" / f"{name}.csv"
        dest.parent.mkdir(parents=True, exist_ok=True)
        # Append batches into one readable CSV per logical table.
        with dest.open("ab") as stream:
            f.write_csv(stream, include_header=dest.stat().st_size == 0)
        self.elapsed[name] = self.elapsed.get(name, 0) + time.perf_counter() - tick
        if name == "transactions":
            self.source_quality(f, day)

    def source_quality(self, f, day):
        r = self.w.r("source-quality", day)
        take = np.flatnonzero(r.random(len(f)) < 0.003)
        if not len(take):
            return
        sample = f[take]
        n = len(sample)
        issue = r.integers(0, 5, n)
        # This quarantined source extract deliberately violates source quality, not canonical integrity.
        sample = sample.with_columns(
            pl.Series(
                "source_issue",
                np.array(
                    [
                        "missing_price",
                        "negative_quantity",
                        "category_alias",
                        "clock_skew",
                        "duplicate_delivery",
                    ]
                )[issue],
            ),
            pl.Series("source_record_id", [f"{day}:{i}" for i in range(n)]),
        )
        sample = sample.with_columns(
            pl.when(pl.col("source_issue") == "missing_price")
            .then(None)
            .otherwise(pl.col("unit_price"))
            .alias("unit_price"),
            pl.when(pl.col("source_issue") == "negative_quantity")
            .then(-1)
            .otherwise(pl.col("quantity"))
            .alias("quantity"),
            pl.when(pl.col("source_issue") == "category_alias")
            .then(pl.lit("mpesa"))
            .otherwise(pl.col("payment_method"))
            .alias("payment_method"),
            pl.when(pl.col("source_issue") == "clock_skew")
            .then(pl.col("transaction_ts") + pl.duration(days=400))
            .otherwise(pl.col("transaction_ts"))
            .alias("transaction_ts"),
        )
        dup = sample.filter(pl.col("source_issue") == "duplicate_delivery")
        sample = pl.concat([sample, dup])
        path = self.root / "source_quality"
        path.mkdir(parents=True, exist_ok=True)
        target = path / "transactions.csv"
        with target.open("ab") as stream:
            sample.write_csv(stream, include_header=target.stat().st_size == 0)
        for key, count in sample.group_by("source_issue").len().iter_rows():
            self.source_stats[key] = self.source_stats.get(key, 0) + count

    def close(self):
        for writer in self.open.values():
            writer.close()
        self.open.clear()

    def report(self):
        self.close()
        con = duckdb.connect()
        con.execute("SET memory_limit='2GB'")
        con.execute("SET threads=4")
        con.execute(f"SET temp_directory='{self.root}/.validation_tmp'")
        for name in self.schema:
            path = str(self.root / "raw" / f"{name}.csv")
            con.execute(
                f"CREATE VIEW {name} AS SELECT * FROM read_csv_auto('{path}')"
            )
        checks = global_checks(con, self.w)
        report = {}
        expected = {
            "customers": self.w.cfg.count("customers"),
            "products_services": self.w.cfg.count("products"),
            "employees": self.w.cfg.count("employees"),
            "transactions": self.w.cfg.count("transactions"),
            "interactions": self.w.cfg.count("interactions"),
            "subscriptions": self.w.cfg.count("subscriptions"),
            "marketing_campaigns": self.w.cfg.count("campaigns"),
            "customer_product_usage": self.w.cfg.count("usage"),
            "sales_pipeline": self.w.cfg.count("pipeline"),
            "support_cases": self.w.cfg.count("support"),
            "invoices_payments": self.w.cfg.count("transactions"),
            "daily_operations": 4380,
            "external_factors": 4380,
        }
        for name, schema in self.schema.items():
            rows = con.execute(f"SELECT count(*) FROM {name}").fetchone()[0]
            if name in expected:
                assert rows == expected[name], (name, rows, expected[name])
            if name == "inventory_movements" and self.w.cfg.scale == 1:
                assert rows >= 2400000, rows
            expr = []
            fields = []
            for col, typ in schema.items():
                q = f'"{col}"'
                expr += [f"count(*)-count({q})", f"approx_count_distinct({q})"]
                fields += [(col, "null_count"), (col, "approx_distinct")]
                if typ.is_numeric():
                    expr += [f"min({q})", f"max({q})", f"avg({q})", f"stddev_pop({q})"]
                    fields += [(col, x) for x in ["min", "max", "mean", "stddev"]]
                elif typ == pl.Date or isinstance(typ, pl.Datetime):
                    expr += [f"min({q})", f"max({q})"]
                    fields += [(col, "min_date"), (col, "max_date")]
            values = con.execute(f'SELECT {",".join(expr)} FROM {name}').fetchone()
            cols = {col: {} for col in schema}
            for (col, metric), value in zip(fields, values):
                cols[col][metric] = value
            for col in cols:
                cols[col]["null_rate"] = cols[col]["null_count"] / rows
                cols[col]["approx_uniqueness_rate"] = min(
                    1, cols[col]["approx_distinct"] / rows
                )
            pk = PK.get(name)
            if pk:
                unique = con.execute(
                    f"SELECT count(DISTINCT {pk}) FROM {name}"
                ).fetchone()[0]
                assert unique == rows
                cols[pk]["exact_uniqueness_rate"] = unique / rows
            else:
                unique = con.execute(
                    f"SELECT count(DISTINCT (date,region)) FROM {name}"
                ).fetchone()[0]
                assert unique == rows
            report[name] = dict(
                dataset=name,
                row_count=rows,
                column_count=len(schema),
                columns=cols,
                duplicate_primary_key_rate=0,
                generation_time_seconds=self.elapsed.get(name, 0),
                generation_time_definition="CPU/wall time in batch validation and CSV writing; simulation times reported separately",
                anomaly_exposure_counts={
                    k: v[name] for k, v in self.w.ep.counts.items() if name in v
                },
            )
        # Distribution diagnostics are empirical, never predetermined analytical answers.
        diagnostics = {
            "transactions_by_region": con.execute(
                "SELECT region,count(*),sum(net_amount) FROM transactions GROUP BY 1 ORDER BY 1"
            ).fetchall(),
            "transactions_by_segment": con.execute(
                "SELECT c.segment,count(*),avg(t.quantity) FROM transactions t JOIN customers c USING(customer_id) GROUP BY 1 ORDER BY 1"
            ).fetchall(),
            "subscription_status": con.execute(
                "SELECT status,count(*) FROM subscriptions GROUP BY 1 ORDER BY 1"
            ).fetchall(),
            "monthly_support": con.execute(
                "SELECT date_trunc('month',opened_at),count(*),avg(resolution_time) FROM support_cases GROUP BY 1 ORDER BY 1"
            ).fetchall(),
            "operations_correlations": con.execute(
                "SELECT corr(backlog_units,processing_time_minutes),corr(processing_time_minutes,SLA_breach_rate),corr(workload_units,overtime_hours) FROM daily_operations"
            ).fetchone(),
            "payment_delay_by_segment": con.execute(
                "SELECT c.segment,avg(i.days_to_payment) FROM invoices_payments i JOIN customers c USING(customer_id) GROUP BY 1 ORDER BY 1"
            ).fetchall(),
        }
        assert len(diagnostics["transactions_by_region"]) == 6
        assert len(diagnostics["subscription_status"]) >= 2
        con.close()
        return dict(
            datasets=report,
            validation_checks=checks,
            distribution_diagnostics=diagnostics,
            source_quality_counts=self.source_stats,
        )

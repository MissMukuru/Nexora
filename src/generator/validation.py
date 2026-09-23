"""Fail-closed chunk contracts and cross-table validation on private staged output."""

import numpy as np
import polars as pl

PK = {
    "customers": "customer_id",
    "products_services": "product_id",
    "employees": "employee_id",
    "transactions": "transaction_id",
    "interactions": "interaction_id",
    "subscriptions": "subscription_id",
    "marketing_campaigns": "campaign_id",
    "customer_product_usage": "usage_id",
    "sales_pipeline": "opportunity_id",
    "support_cases": "case_id",
    "invoices_payments": "invoice_id",
    "inventory_movements": "movement_id",
    "subscription_events": "subscription_event_id",
    "pipeline_stage_history": "pipeline_event_id",
    "suppliers": "supplier_id",
}


class Validator:
    def __init__(self, w):
        self.w = w
        self.last = {}
        self.rows = {}

    def validate(self, name, f):
        if not f.height:
            return
        w = self.w
        date_columns = {
            "transactions": "transaction_ts",
            "interactions": "interaction_ts",
            "customer_product_usage": "date",
            "daily_operations": "date",
            "external_factors": "date",
            "inventory_movements": "date",
            "support_cases": "opened_at",
            "sales_pipeline": "created_at",
            "pipeline_stage_history": "event_date",
            "subscription_events": "event_date",
            "invoices_payments": "invoice_date",
            "subscriptions": "start_date",
            "marketing_campaigns": "start_date",
        }
        if name in date_columns:
            values = f[date_columns[name]].to_numpy().astype("datetime64[D]")
            assert np.all(
                (values >= np.datetime64("2025-01-01"))
                & (values <= np.datetime64("2026-12-31"))
            ), f"{name}: observation dates out of range"
        domains = {
            "segment": ["Micro", "SMB", "Midmarket", "Enterprise"],
            "category": ["Software", "Hardware", "Consulting", "Training", "Support"],
            "currency": ["KES"],
            "payment_method": ["M-Pesa", "Bank transfer", "Card", "Credit account"],
            "sales_channel": [
                "Search",
                "Social",
                "Email",
                "Partner",
                "Events",
                "Direct",
            ],
            "stage": [
                "Qualified",
                "Discovery",
                "Proposal",
                "Negotiation",
                "Closed Won",
                "Closed Lost",
            ],
        }
        for col, domain in domains.items():
            if col in f.columns:
                assert (
                    f[col].drop_nulls().is_in(domain).all()
                ), f"{name}.{col}: unexpected category"
        if name in PK:
            a = f[PK[name]].to_numpy()
            if not (np.all(np.diff(a) > 0) and a[0] > self.last.get(name, 0)):
                raise ValueError(f"{name}: duplicate/non-monotonic PK")
            self.last[name] = int(a[-1])
        else:
            assert f.select(pl.struct("date", "region").n_unique()).item() == f.height
        limits = {
            "customer_id": w.cfg.count("customers"),
            "product_id": w.cfg.count("products"),
            "employee_id": w.cfg.count("employees"),
            "sales_rep_id": w.cfg.count("employees"),
            "campaign_id": w.cfg.count("campaigns"),
            "subscription_id": w.cfg.count("subscriptions"),
            "opportunity_id": w.cfg.count("pipeline"),
            "supplier_id": 80,
            "transaction_id": w.ids.get("transactions", 0),
            "case_id": w.ids.get("support_cases", 0),
        }
        for col, limit in limits.items():
            if col in f.columns and col != PK.get(name):
                a = f[col].drop_nulls().to_numpy()
                assert np.all((a >= 1) & (a <= limit)), f"{name}.{col}: FK out of range"
        for col in f.columns:
            s = f[col]
            if s.dtype.is_numeric():
                assert not s.is_nan().any(), f"{name}.{col}: NaN"
                assert not s.is_infinite().any(), f"{name}.{col}: inf"
        for col in [
            "discount_rate",
            "margin_target",
            "feature_adoption_rate",
            "error_rate",
            "probability",
            "SLA_breach_rate",
            "absenteeism_rate",
            "conversion_rate",
        ]:
            if col in f.columns:
                assert (
                    f[col].drop_nulls().is_between(0, 1).all()
                ), f"{name}.{col}: proportion"
        for col in [
            "quantity",
            "unit_price",
            "gross_amount",
            "discount_amount",
            "net_amount",
            "cost_amount",
            "monthly_price",
            "workload_units",
            "backlog_units",
            "paid_amount",
            "invoice_amount",
            "stock_level_after",
            "usage_events",
            "active_users",
            "usage_minutes",
            "resolution_time",
            "first_response_time",
            "tenure_days",
            "overdue_days",
            "days_to_payment",
        ]:
            if col in f.columns and not (
                name == "inventory_movements" and col == "quantity"
            ):
                assert (f[col].drop_nulls() >= 0).all(), f"{name}.{col}: negative"
        if "region" in f.columns:
            assert (
                f["region"]
                .is_in(["Nairobi", "Kiambu", "Nakuru", "Eldoret", "Mombasa", "Kisumu"])
                .all()
            )
        if "satisfaction_score" in f.columns:
            assert f["satisfaction_score"].drop_nulls().is_between(1, 5).all()
        if "sentiment_score" in f.columns:
            assert f["sentiment_score"].is_between(-1, 1).all()
        if name == "transactions":
            for lhs, rhs in [
                ("gross_amount", f["quantity"] * f["unit_price"]),
                ("discount_amount", (f["gross_amount"] * f["discount_rate"]).round(2)),
                ("net_amount", f["gross_amount"] - f["discount_amount"]),
                ("gross_margin", f["net_amount"] - f["cost_amount"]),
            ]:
                assert np.allclose(
                    f[lhs].to_numpy(), rhs.to_numpy(), atol=0.011, rtol=0
                ), lhs
            assert f["transaction_status"].is_in(["Completed", "Backordered"]).all()
        if name == "subscriptions":
            assert (f["renewal_date"] >= f["start_date"]).all()
            assert f.filter(
                pl.col("end_date").is_not_null()
                & (pl.col("end_date") < pl.col("start_date"))
            ).is_empty()
            assert f.filter(
                (pl.col("status") == "Active") & pl.col("end_date").is_not_null()
            ).is_empty()
            assert f.filter(
                (pl.col("status") != "Active") & pl.col("end_date").is_null()
            ).is_empty()
        if name == "support_cases":
            assert f.filter(pl.col("closed_at") < pl.col("opened_at")).is_empty()
        if name == "invoices_payments":
            assert (f["paid_amount"] <= f["invoice_amount"] + 0.001).all()
            assert (f["due_date"] >= f["invoice_date"]).all()
            assert f.filter(pl.col("payment_date") < pl.col("invoice_date")).is_empty()
        if name == "marketing_campaigns":
            for high, low in [
                ("impressions", "clicks"),
                ("clicks", "leads"),
                ("leads", "qualified_leads"),
                ("qualified_leads", "conversions"),
            ]:
                assert (f[high] >= f[low]).all()
            assert np.allclose(
                f["conversion_rate"], f["conversions"] / f["leads"].clip(1)
            )
            assert np.allclose(
                f["roi"], (f["revenue"] - f["spend"]) / f["spend"].clip(0.01)
            )
        if name == "customer_product_usage":
            a = w.sub
            ix = f["subscription_id"].to_numpy() - 1
            assert np.all(a["active"][ix])
            assert np.all(a["c"][ix] + 1 == f["customer_id"].to_numpy())
            assert np.all(a["p"][ix] + 1 == f["product_id"].to_numpy())
        self.rows[name] = self.rows.get(name, 0) + f.height


def global_checks(con, w):
    checks = {}

    def zero(name, sql):
        count = int(con.execute(sql).fetchone()[0])
        checks[name] = count
        if count:
            raise ValueError(f"{name}: {count} violations")

    zero(
        "transaction_customer_region",
        """SELECT count(*) FROM transactions t JOIN customers c USING(customer_id) WHERE t.region<>c.region OR CAST(t.transaction_ts AS DATE)<c.acquisition_date""",
    )
    zero(
        "case_order_ownership",
        """SELECT count(*) FROM support_cases s JOIN transactions t USING(transaction_id) WHERE s.customer_id<>t.customer_id OR s.product_id<>t.product_id OR s.opened_at<t.transaction_ts""",
    )
    zero(
        "interaction_case_customer",
        """SELECT count(*) FROM interactions i JOIN support_cases s USING(case_id) WHERE i.customer_id<>s.customer_id OR i.employee_id<>s.employee_id""",
    )
    zero(
        "invoice_order_customer_amount",
        """SELECT count(*) FROM invoices_payments i JOIN transactions t USING(transaction_id) WHERE i.customer_id<>t.customer_id OR abs(i.invoice_amount-t.net_amount)>.011""",
    )
    zero(
        "opportunity_order_provenance",
        """SELECT count(*) FROM transactions t JOIN sales_pipeline p USING(opportunity_id) WHERE t.customer_id<>p.customer_id OR t.product_id<>p.product_id OR t.campaign_id<>p.campaign_id OR p.stage<>'Closed Won' OR CAST(t.transaction_ts AS DATE)<>p.actual_close_date""",
    )
    zero(
        "campaign_target_provenance",
        """SELECT count(*) FROM sales_pipeline p JOIN marketing_campaigns m USING(campaign_id) JOIN customers c USING(customer_id) WHERE p.product_id<>m.product_id OR c.region<>m.target_region OR c.segment<>m.target_segment OR CAST(p.created_at AS DATE)<m.start_date OR CAST(p.created_at AS DATE)>m.end_date""",
    )
    zero(
        "won_opportunity_has_transaction",
        """SELECT count(*) FROM sales_pipeline p LEFT JOIN transactions t USING(opportunity_id) WHERE p.stage='Closed Won' AND t.transaction_id IS NULL""",
    )
    zero(
        "campaign_revenue_reconciliation",
        """SELECT count(*) FROM marketing_campaigns m LEFT JOIN (SELECT campaign_id,sum(net_amount) revenue,count(*) conversions FROM transactions WHERE transaction_status='Completed' AND campaign_id IS NOT NULL GROUP BY 1) t USING(campaign_id) WHERE abs(m.revenue-coalesce(t.revenue,0))>.011 OR m.conversions<>coalesce(t.conversions,0)""",
    )
    zero(
        "inventory_ledger_prefix",
        """SELECT count(*) FROM (SELECT stock_level_after, sum(quantity) OVER(PARTITION BY product_id,region ORDER BY movement_id ROWS UNBOUNDED PRECEDING) balance FROM inventory_movements) WHERE balance<>stock_level_after OR balance<0""",
    )
    zero(
        "inventory_shipment_matches_order",
        """SELECT count(*) FROM inventory_movements m JOIN transactions t USING(transaction_id) WHERE m.movement_type='Sale' AND (m.product_id<>t.product_id OR m.region<>t.region OR -m.quantity<>t.quantity OR t.transaction_status<>'Completed')""",
    )
    zero(
        "completed_hardware_has_shipment",
        """SELECT count(*) FROM transactions t JOIN products_services p USING(product_id) LEFT JOIN inventory_movements m USING(transaction_id) WHERE p.category='Hardware' AND t.transaction_status='Completed' AND m.movement_id IS NULL""",
    )
    zero(
        "lifetime_value_reconciliation",
        """SELECT count(*) FROM customers c LEFT JOIN (SELECT customer_id,sum(net_amount) net FROM transactions WHERE transaction_status='Completed' GROUP BY 1) t USING(customer_id) WHERE abs(c.customer_lifetime_value-coalesce(t.net,0))>.011""",
    )
    zero(
        "subscription_terminal_state",
        """WITH terminal AS (SELECT subscription_id,arg_max(event_type,subscription_event_id) last_kind FROM subscription_events WHERE event_type IN ('Started','Ended','Reactivated') GROUP BY 1) SELECT count(*) FROM subscriptions s JOIN terminal t USING(subscription_id) WHERE (s.status='Active')<>(t.last_kind<>'Ended')""",
    )
    zero(
        "case_resolution_duration",
        """SELECT count(*) FROM support_cases WHERE status='Resolved' AND abs(epoch(closed_at-opened_at)/3600-resolution_time)>.002""",
    )
    zero(
        "invoice_payment_timing",
        """SELECT count(*) FROM invoices_payments WHERE payment_date>DATE '2026-12-31' OR (payment_date IS NOT NULL AND date_diff('day',invoice_date,payment_date)<>days_to_payment)""",
    )
    zero(
        "subscription_event_dates",
        """SELECT count(*) FROM subscription_events e JOIN subscriptions s USING(subscription_id) WHERE e.event_date<s.start_date OR e.customer_id<>s.customer_id OR e.product_id<>s.product_id""",
    )
    # All foreign keys: relational check, not just min/max ranges.
    parents = {
        "customer_id": ("customers", "customer_id"),
        "product_id": ("products_services", "product_id"),
        "employee_id": ("employees", "employee_id"),
        "sales_rep_id": ("employees", "employee_id"),
        "campaign_id": ("marketing_campaigns", "campaign_id"),
        "subscription_id": ("subscriptions", "subscription_id"),
        "opportunity_id": ("sales_pipeline", "opportunity_id"),
        "supplier_id": ("suppliers", "supplier_id"),
        "transaction_id": ("transactions", "transaction_id"),
        "case_id": ("support_cases", "case_id"),
    }
    tables = [x[0] for x in con.execute("SHOW TABLES").fetchall()]
    for table in tables:
        cols = [x[0] for x in con.execute(f'DESCRIBE "{table}"').fetchall()]
        for col, (parent, key) in parents.items():
            if col in cols and table != parent:
                zero(
                    f"fk_{table}_{col}",
                    f'SELECT count(*) FROM "{table}" t ANTI JOIN "{parent}" p ON t."{col}"=p."{key}" WHERE t."{col}" IS NOT NULL',
                )
    return checks

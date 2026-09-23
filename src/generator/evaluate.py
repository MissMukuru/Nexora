"""Evaluator-only sanity checks for intended interventions; never an agent input.

These are diagnostics, not causal estimators: a real investigation must account
for overlapping shocks, selection, mix, trend and seasonality.
"""

import argparse
import json
from pathlib import Path
import duckdb

QUERIES = {
    "operational_window": """SELECT region,
        CASE WHEN date BETWEEN DATE '2026-03-15' AND DATE '2026-04-15' THEN 'during' ELSE 'before' END period,
        avg(processing_time_minutes),avg(backlog_units),avg(downtime_minutes),avg(absenteeism_rate)
        FROM daily_operations WHERE date BETWEEN DATE '2026-02-11' AND DATE '2026-04-15'
        GROUP BY 1,2 ORDER BY 1,2""",
    "software_normalized_prices": """SELECT CASE WHEN transaction_ts< TIMESTAMP '2026-04-01' THEN 'before' ELSE 'after' END period,
        median(unit_price/base_price) price_ratio,count(*)
        FROM transactions JOIN products_services USING(product_id)
        WHERE category='Software' AND transaction_ts>=TIMESTAMP '2026-03-01' AND transaction_ts<TIMESTAMP '2026-05-01'
        GROUP BY 1 ORDER BY 1""",
    "marketing_funnel_window": """SELECT month(start_date) AS month_number,avg(spend),avg(leads),
        sum(qualified_leads)*1.0/sum(leads) qualification_rate,
        sum(conversions)*1.0/sum(leads) conversion_rate,sum(revenue)/sum(spend) revenue_per_spend
        FROM marketing_campaigns WHERE start_date>=DATE '2026-04-01' AND start_date<DATE '2026-07-01'
        GROUP BY 1 ORDER BY 1""",
    "regional_economic_window": """SELECT region,month(date),avg(fuel_price),avg(inflation_rate),avg(business_confidence)
        FROM external_factors WHERE date>=DATE '2026-05-01' AND date<DATE '2026-07-01'
        GROUP BY 1,2 ORDER BY 1,2""",
    "regional_fulfilled_revenue": """SELECT region,month(transaction_ts),count(*),sum(net_amount)
        FROM transactions WHERE transaction_status='Completed' AND transaction_ts>=TIMESTAMP '2026-05-01' AND transaction_ts<TIMESTAMP '2026-07-01'
        GROUP BY 1,2 ORDER BY 1,2""",
    "support_window": """SELECT c.region,
        CASE WHEN opened_at>=TIMESTAMP '2026-03-15' THEN 'during' ELSE 'before' END period,
        count(*),avg(resolution_time),avg(satisfaction_score)
        FROM support_cases s JOIN customers c USING(customer_id)
        WHERE opened_at>=TIMESTAMP '2026-02-11' AND opened_at<TIMESTAMP '2026-04-16'
        GROUP BY 1,2 ORDER BY 1,2""",
}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("output", nargs="?", default="data")
    args = p.parse_args()
    root = Path(args.output).resolve()
    assert (root / "_SUCCESS").exists()
    con = duckdb.connect()
    con.execute("SET memory_limit='2GB'")
    con.execute("SET threads=4")
    for table in [
        "daily_operations",
        "transactions",
        "products_services",
        "marketing_campaigns",
        "external_factors",
        "support_cases",
        "customers",
    ]:
        safe = str(root).replace("'", "''")
        con.execute(
            f"CREATE VIEW {table} AS SELECT * FROM read_csv_auto('{safe}/raw/{table}.csv')"
        )
    result = {}
    for name, sql in QUERIES.items():
        cur = con.execute(sql)
        names = [d[0] for d in cur.description]
        result[name] = [dict(zip(names, row)) for row in cur.fetchall()]
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()

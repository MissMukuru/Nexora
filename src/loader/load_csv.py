"""Bulk-load the generated flat CSV tables into PostgreSQL."""

import argparse
import csv
import os
from pathlib import Path

import psycopg
from psycopg import sql

TABLES = {
    "customers": {
        "customer_id": "INTEGER PRIMARY KEY",
        "company_name": "TEXT NOT NULL",
        "segment": "TEXT NOT NULL",
        "region": "TEXT NOT NULL",
        "industry": "TEXT NOT NULL",
        "company_size": "TEXT NOT NULL",
        "employee_count": "INTEGER NOT NULL",
        "annual_revenue_band": "TEXT",
        "acquisition_date": "DATE NOT NULL",
        "acquisition_channel": "TEXT",
        "customer_lifetime_value": "NUMERIC(16,2) NOT NULL",
        "account_status": "TEXT NOT NULL",
    },
    "products_services": {
        "product_id": "INTEGER PRIMARY KEY",
        "product_name": "TEXT NOT NULL",
        "category": "TEXT NOT NULL",
        "business_unit": "TEXT NOT NULL",
        "product_family": "TEXT NOT NULL",
        "cost": "NUMERIC(16,2) NOT NULL",
        "base_price": "NUMERIC(16,2) NOT NULL",
        "launch_date": "DATE NOT NULL",
        "lifecycle_stage": "TEXT NOT NULL",
        "margin_target": "NUMERIC(8,6) NOT NULL",
        "recurring_flag": "BOOLEAN NOT NULL",
    },
    "employees": {
        "employee_id": "INTEGER PRIMARY KEY",
        "employee_name": "TEXT NOT NULL",
        "department": "TEXT NOT NULL",
        "job_level": "TEXT NOT NULL",
        "region": "TEXT NOT NULL",
        "contract_type": "TEXT NOT NULL",
        "hire_date": "DATE NOT NULL",
        "annual_salary": "NUMERIC(16,2) NOT NULL",
        "performance_band": "TEXT NOT NULL",
        "tenure_months": "INTEGER NOT NULL",
    },
    "suppliers": {
        "supplier_id": "INTEGER PRIMARY KEY",
        "supplier_name": "TEXT NOT NULL",
        "supplier_region": "TEXT NOT NULL",
        "base_lead_time_days": "INTEGER NOT NULL",
    },
    "marketing_campaigns": {
        "campaign_id": "INTEGER PRIMARY KEY",
        "campaign_name": "TEXT NOT NULL",
        "channel": "TEXT NOT NULL",
        "campaign_type": "TEXT NOT NULL",
        "target_segment": "TEXT NOT NULL",
        "target_region": "TEXT NOT NULL",
        "product_id": "INTEGER NOT NULL",
        "start_date": "DATE NOT NULL",
        "end_date": "DATE NOT NULL",
        "budget": "NUMERIC(16,2) NOT NULL",
        "spend": "NUMERIC(16,2) NOT NULL",
        "impressions": "BIGINT NOT NULL",
        "clicks": "BIGINT NOT NULL",
        "leads": "BIGINT NOT NULL",
        "qualified_leads": "BIGINT NOT NULL",
        "conversions": "BIGINT NOT NULL",
        "conversion_rate": "DOUBLE PRECISION NOT NULL",
        "customer_acquisition_cost": "NUMERIC(16,2)",
        "revenue": "NUMERIC(16,2) NOT NULL",
        "roi": "DOUBLE PRECISION NOT NULL",
    },
    "sales_pipeline": {
        "opportunity_id": "BIGINT PRIMARY KEY",
        "customer_id": "INTEGER NOT NULL",
        "sales_rep_id": "INTEGER NOT NULL",
        "product_id": "INTEGER NOT NULL",
        "campaign_id": "INTEGER",
        "created_at": "TIMESTAMP NOT NULL",
        "expected_close_date": "DATE NOT NULL",
        "actual_close_date": "DATE",
        "stage": "TEXT NOT NULL",
        "probability": "DOUBLE PRECISION NOT NULL",
        "deal_value": "NUMERIC(16,2) NOT NULL",
        "lost_reason": "TEXT",
        "source": "TEXT NOT NULL",
        "region": "TEXT NOT NULL",
    },
    "subscriptions": {
        "subscription_id": "BIGINT PRIMARY KEY",
        "customer_id": "INTEGER NOT NULL",
        "product_id": "INTEGER NOT NULL",
        "plan": "TEXT NOT NULL",
        "start_date": "DATE NOT NULL",
        "renewal_date": "DATE NOT NULL",
        "end_date": "DATE",
        "status": "TEXT NOT NULL",
        "monthly_price": "NUMERIC(16,2) NOT NULL",
        "discount_rate": "DOUBLE PRECISION NOT NULL",
        "cancellation_reason": "TEXT",
        "auto_renew": "BOOLEAN NOT NULL",
        "tenure_days": "INTEGER NOT NULL",
    },
    "transactions": {
        "transaction_id": "BIGINT PRIMARY KEY",
        "customer_id": "INTEGER NOT NULL",
        "transaction_ts": "TIMESTAMP NOT NULL",
        "product_id": "INTEGER NOT NULL",
        "quantity": "INTEGER NOT NULL",
        "unit_price": "NUMERIC(16,2) NOT NULL",
        "gross_amount": "NUMERIC(16,2) NOT NULL",
        "discount_rate": "DOUBLE PRECISION NOT NULL",
        "discount_amount": "NUMERIC(16,2) NOT NULL",
        "net_amount": "NUMERIC(16,2) NOT NULL",
        "cost_amount": "NUMERIC(16,2) NOT NULL",
        "gross_margin": "NUMERIC(16,2) NOT NULL",
        "payment_method": "TEXT NOT NULL",
        "sales_channel": "TEXT NOT NULL",
        "sales_rep_id": "INTEGER",
        "region": "TEXT NOT NULL",
        "currency": "TEXT NOT NULL",
        "transaction_status": "TEXT NOT NULL",
        "campaign_id": "INTEGER",
        "opportunity_id": "BIGINT",
        "ingested_at": "TIMESTAMP",
    },
    "support_cases": {
        "case_id": "BIGINT PRIMARY KEY",
        "customer_id": "INTEGER NOT NULL",
        "product_id": "INTEGER NOT NULL",
        "transaction_id": "BIGINT NOT NULL",
        "employee_id": "INTEGER NOT NULL",
        "opened_at": "TIMESTAMP NOT NULL",
        "closed_at": "TIMESTAMP",
        "issue_category": "TEXT NOT NULL",
        "severity": "INTEGER NOT NULL",
        "status": "TEXT NOT NULL",
        "escalation_flag": "BOOLEAN NOT NULL",
        "resolution_time": "DOUBLE PRECISION",
        "satisfaction_score": "DOUBLE PRECISION",
        "ingested_at": "TIMESTAMP",
    },
    "interactions": {
        "interaction_id": "BIGINT PRIMARY KEY",
        "customer_id": "INTEGER NOT NULL",
        "employee_id": "INTEGER NOT NULL",
        "case_id": "BIGINT",
        "interaction_ts": "TIMESTAMP NOT NULL",
        "interaction_type": "TEXT NOT NULL",
        "channel": "TEXT NOT NULL",
        "priority": "TEXT NOT NULL",
        "issue_category": "TEXT NOT NULL",
        "status": "TEXT NOT NULL",
        "sentiment_score": "DOUBLE PRECISION",
        "first_response_time": "DOUBLE PRECISION NOT NULL",
        "resolution_time": "DOUBLE PRECISION",
        "escalation_flag": "BOOLEAN NOT NULL",
        "satisfaction_score": "DOUBLE PRECISION",
        "region": "TEXT NOT NULL",
        "ingested_at": "TIMESTAMP",
    },
    "invoices_payments": {
        "invoice_id": "BIGINT PRIMARY KEY",
        "customer_id": "INTEGER NOT NULL",
        "transaction_id": "BIGINT NOT NULL",
        "invoice_date": "DATE NOT NULL",
        "due_date": "DATE NOT NULL",
        "payment_date": "DATE",
        "invoice_amount": "NUMERIC(16,2) NOT NULL",
        "paid_amount": "NUMERIC(16,2) NOT NULL",
        "payment_status": "TEXT NOT NULL",
        "days_to_payment": "INTEGER",
        "overdue_days": "INTEGER NOT NULL",
        "ingested_at": "TIMESTAMP",
    },
    "customer_product_usage": {
        "usage_id": "BIGINT PRIMARY KEY",
        "subscription_id": "BIGINT NOT NULL",
        "customer_id": "INTEGER NOT NULL",
        "product_id": "INTEGER NOT NULL",
        "date": "DATE NOT NULL",
        "usage_events": "INTEGER NOT NULL",
        "active_users": "INTEGER NOT NULL",
        "usage_minutes": "DOUBLE PRECISION NOT NULL",
        "feature_adoption_rate": "DOUBLE PRECISION",
        "error_rate": "DOUBLE PRECISION NOT NULL",
        "ingested_at": "TIMESTAMP",
    },
    "daily_operations": {
        "date": "DATE NOT NULL",
        "region": "TEXT NOT NULL",
        "workload_units": "DOUBLE PRECISION NOT NULL",
        "completed_units": "DOUBLE PRECISION NOT NULL",
        "backlog_units": "DOUBLE PRECISION NOT NULL",
        "opening_backlog_units": "DOUBLE PRECISION NOT NULL",
        "processing_time_minutes": "DOUBLE PRECISION NOT NULL",
        "downtime_minutes": "DOUBLE PRECISION NOT NULL",
        "incident_count": "INTEGER NOT NULL",
        "staffing_level": "INTEGER NOT NULL",
        "scheduled_staffing": "INTEGER NOT NULL",
        "absenteeism_rate": "DOUBLE PRECISION NOT NULL",
        "overtime_hours": "DOUBLE PRECISION NOT NULL",
        "sla_breach_rate": "DOUBLE PRECISION NOT NULL",
        "operating_cost": "NUMERIC(16,2) NOT NULL",
        "ingested_at": "TIMESTAMP",
    },
    "external_factors": {
        "date": "DATE NOT NULL",
        "region": "TEXT NOT NULL",
        "fuel_price": "DOUBLE PRECISION NOT NULL",
        "inflation_rate": "DOUBLE PRECISION NOT NULL",
        "business_confidence": "DOUBLE PRECISION NOT NULL",
        "exchange_rate": "DOUBLE PRECISION NOT NULL",
        "rainfall_index": "DOUBLE PRECISION NOT NULL",
        "traffic_index": "DOUBLE PRECISION NOT NULL",
        "market_activity_index": "DOUBLE PRECISION NOT NULL",
        "holiday_flag": "BOOLEAN NOT NULL",
    },
    "inventory_movements": {
        "movement_id": "BIGINT PRIMARY KEY",
        "product_id": "INTEGER NOT NULL",
        "region": "TEXT NOT NULL",
        "date": "DATE NOT NULL",
        "movement_type": "TEXT NOT NULL",
        "quantity": "INTEGER NOT NULL",
        "unit_cost": "NUMERIC(16,2) NOT NULL",
        "supplier_id": "INTEGER NOT NULL",
        "lead_time_days": "INTEGER NOT NULL",
        "stock_level_after": "INTEGER NOT NULL",
        "transaction_id": "BIGINT",
        "ingested_at": "TIMESTAMP",
    },
    "subscription_events": {
        "subscription_event_id": "BIGINT PRIMARY KEY",
        "subscription_id": "BIGINT NOT NULL",
        "customer_id": "INTEGER NOT NULL",
        "product_id": "INTEGER NOT NULL",
        "event_date": "DATE NOT NULL",
        "event_type": "TEXT NOT NULL",
        "old_plan": "INTEGER NOT NULL",
        "new_plan": "INTEGER NOT NULL",
        "old_monthly_price": "NUMERIC(16,2) NOT NULL",
        "new_monthly_price": "NUMERIC(16,2) NOT NULL",
        "ingested_at": "TIMESTAMP",
    },
    "pipeline_stage_history": {
        "pipeline_event_id": "BIGINT PRIMARY KEY",
        "opportunity_id": "BIGINT NOT NULL",
        "event_date": "DATE NOT NULL",
        "stage": "TEXT NOT NULL",
    },
}

LOAD_ORDER = [
    "customers", "products_services", "employees", "suppliers",
    "marketing_campaigns", "sales_pipeline", "subscriptions",
    "transactions", "support_cases", "interactions", "invoices_payments",
    "customer_product_usage", "daily_operations", "external_factors",
    "inventory_movements", "subscription_events", "pipeline_stage_history",
]

FOREIGN_KEYS = [
    ("marketing_campaigns", "product_id", "products_services", "product_id"),
    ("sales_pipeline", "customer_id", "customers", "customer_id"),
    ("sales_pipeline", "sales_rep_id", "employees", "employee_id"),
    ("sales_pipeline", "product_id", "products_services", "product_id"),
    ("sales_pipeline", "campaign_id", "marketing_campaigns", "campaign_id"),
    ("subscriptions", "customer_id", "customers", "customer_id"),
    ("subscriptions", "product_id", "products_services", "product_id"),
    ("transactions", "customer_id", "customers", "customer_id"),
    ("transactions", "product_id", "products_services", "product_id"),
    ("transactions", "sales_rep_id", "employees", "employee_id"),
    ("transactions", "campaign_id", "marketing_campaigns", "campaign_id"),
    ("transactions", "opportunity_id", "sales_pipeline", "opportunity_id"),
    ("support_cases", "customer_id", "customers", "customer_id"),
    ("support_cases", "product_id", "products_services", "product_id"),
    ("support_cases", "transaction_id", "transactions", "transaction_id"),
    ("support_cases", "employee_id", "employees", "employee_id"),
    ("interactions", "customer_id", "customers", "customer_id"),
    ("interactions", "employee_id", "employees", "employee_id"),
    ("interactions", "case_id", "support_cases", "case_id"),
    ("invoices_payments", "customer_id", "customers", "customer_id"),
    ("invoices_payments", "transaction_id", "transactions", "transaction_id"),
    ("customer_product_usage", "subscription_id", "subscriptions", "subscription_id"),
    ("customer_product_usage", "customer_id", "customers", "customer_id"),
    ("customer_product_usage", "product_id", "products_services", "product_id"),
    ("inventory_movements", "product_id", "products_services", "product_id"),
    ("inventory_movements", "supplier_id", "suppliers", "supplier_id"),
    ("inventory_movements", "transaction_id", "transactions", "transaction_id"),
    ("subscription_events", "subscription_id", "subscriptions", "subscription_id"),
    ("subscription_events", "customer_id", "customers", "customer_id"),
    ("subscription_events", "product_id", "products_services", "product_id"),
    ("pipeline_stage_history", "opportunity_id", "sales_pipeline", "opportunity_id"),
]


def connection_string():
    return os.getenv(
        "NEXORA_DATABASE_URL",
        "host=localhost port=5433 dbname=nexora user=nexora password=nexora",
    )


def create_tables(conn):
    with conn.cursor() as cur:
        cur.execute("CREATE SCHEMA IF NOT EXISTS raw")
        for table, columns in TABLES.items():
            definition = sql.SQL(", ").join(
                sql.SQL("{} {}").format(sql.Identifier(column), sql.SQL(type_sql))
                for column, type_sql in columns.items()
            )
            cur.execute(
                sql.SQL("CREATE TABLE IF NOT EXISTS raw.{} ({})").format(
                    sql.Identifier(table), definition
                )
            )


def clear_tables(conn):
    with conn.cursor() as cur:
        for table, column, _, _ in FOREIGN_KEYS:
            cur.execute(sql.SQL(
                "ALTER TABLE raw.{} DROP CONSTRAINT IF EXISTS {}"
            ).format(sql.Identifier(table), sql.Identifier(f"{table}_{column}_fkey")))
        for table, column in [
            ("transactions", "customer_id"), ("transactions", "product_id"),
            ("transactions", "transaction_ts"), ("support_cases", "opened_at"),
            ("invoices_payments", "customer_id"), ("customer_product_usage", "date"),
            ("inventory_movements", "product_id"),
        ]:
            cur.execute(sql.SQL("DROP INDEX IF EXISTS raw.{}").format(
                sql.Identifier(f"{table}_{column}_idx")
            ))
        names = sql.SQL(", ").join(sql.Identifier(table) for table in reversed(LOAD_ORDER))
        cur.execute(sql.SQL("TRUNCATE TABLE {} CASCADE").format(
            sql.SQL(", ").join(sql.SQL("raw.{}").format(sql.Identifier(table)) for table in reversed(LOAD_ORDER))
        ))


def load_csv(conn, table, input_dir):
    path = input_dir / f"{table}.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    columns = list(TABLES[table])
    quoted_columns = sql.SQL(", ").join(sql.Identifier(column) for column in columns)
    statement = sql.SQL(
        "COPY raw.{} ({}) FROM STDIN WITH (FORMAT CSV, HEADER TRUE, NULL '')"
    ).format(sql.Identifier(table), quoted_columns)
    with path.open("r", encoding="utf-8", newline="") as stream:
        with conn.cursor().copy(statement) as copy:
            while chunk := stream.read(1024 * 1024):
                copy.write(chunk)
    with conn.cursor() as cur:
        cur.execute(sql.SQL("SELECT COUNT(*) FROM raw.{}").format(sql.Identifier(table)))
        return cur.fetchone()[0]


def add_constraints_and_indexes(conn):
    with conn.cursor() as cur:
        for table, column, parent, parent_column in FOREIGN_KEYS:
            name = f"{table}_{column}_fkey"
            cur.execute(sql.SQL(
                "ALTER TABLE raw.{} ADD CONSTRAINT {} FOREIGN KEY ({}) REFERENCES raw.{} ({})"
            ).format(
                sql.Identifier(table), sql.Identifier(name), sql.Identifier(column),
                sql.Identifier(parent), sql.Identifier(parent_column),
            ))
        for table, column in [
            ("transactions", "customer_id"), ("transactions", "product_id"),
            ("transactions", "transaction_ts"), ("support_cases", "opened_at"),
            ("invoices_payments", "customer_id"), ("customer_product_usage", "date"),
            ("inventory_movements", "product_id"),
        ]:
            name = f"{table}_{column}_idx"
            cur.execute(sql.SQL("CREATE INDEX {} ON raw.{} ({})").format(
                sql.Identifier(name), sql.Identifier(table), sql.Identifier(column)
            ))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data-csv-flat/raw"))
    parser.add_argument("--replace", action="store_true", help="Clear existing raw rows first")
    args = parser.parse_args()

    with psycopg.connect(connection_string()) as conn:
        create_tables(conn)
        if args.replace:
            clear_tables(conn)
        counts = {table: load_csv(conn, table, args.input) for table in LOAD_ORDER}
        add_constraints_and_indexes(conn)
        conn.commit()

    print("Loaded PostgreSQL raw schema:")
    for table, count in counts.items():
        print(f"  {table}: {count:,} rows")


if __name__ == "__main__":
    main()

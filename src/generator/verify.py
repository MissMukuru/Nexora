"""Reopen completed output and independently rerun relational checks."""

import argparse
import json
from pathlib import Path
import duckdb
from .validation import global_checks, PK


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", nargs="?", default="data")
    args = parser.parse_args()
    root = Path(args.output).resolve()
    assert (root / "_SUCCESS").exists()
    report = json.loads((root / "generation_report.json").read_text())
    con = duckdb.connect()
    con.execute("SET memory_limit='2GB'")
    con.execute("SET threads=4")
    for name, meta in report["datasets"].items():
        con.execute(
            f"CREATE VIEW {name} AS SELECT * FROM read_csv_auto('{root}/raw/{name}.csv')"
        )
        assert (
            con.execute(f"SELECT count(*) FROM {name}").fetchone()[0]
            == meta["row_count"]
        )
        if name in PK:
            assert (
                con.execute(
                    f"SELECT count(DISTINCT {PK[name]}) FROM {name}"
                ).fetchone()[0]
                == meta["row_count"]
            )
    result = global_checks(con, None)
    print(
        json.dumps(
            {
                "rows": report["row_count"],
                "checks": len(result),
                "violations": sum(result.values()),
                "status": "passed",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

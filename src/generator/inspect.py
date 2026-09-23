"""Inspect generated CSV data from the command line."""

import argparse
from pathlib import Path

import duckdb


def table_names(root):
    return sorted(path.stem for path in (root / "raw").glob("*.csv"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", nargs="?", default="data")
    parser.add_argument("--table", help="Table to display")
    parser.add_argument("--limit", type=int, default=10)
    args = parser.parse_args()

    if args.limit < 1:
        parser.error("--limit must be at least 1")

    root = Path(args.output).resolve()
    raw = root / "raw"
    if not raw.exists():
        parser.error(f"No raw data directory found: {raw}")

    names = table_names(root)
    if not names:
        parser.error(f"No CSV tables found under {raw}")

    if args.table is None:
        print("Available tables:")
        for name in names:
            print(f"- {name}")
        return

    if args.table not in names:
        parser.error(f"Unknown table '{args.table}'. Choose from: {', '.join(names)}")

    database = duckdb.connect()
    pattern = str(raw / f"{args.table}.csv").replace("\\", "/")
    rows = database.execute(
        f"SELECT * FROM read_csv_auto('{pattern}') LIMIT ?",
        [args.limit],
    ).fetchdf()
    print(rows.to_string(index=False))


if __name__ == "__main__":
    main()
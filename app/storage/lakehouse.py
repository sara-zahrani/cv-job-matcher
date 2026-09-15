"""
Lakehouse storage.

Clean postings are written to a Delta table: ordinary Parquet files in a
folder, plus a transaction log (_delta_log/) that records every write.
The log is what turns a folder of files into something with database
guarantees: atomic appends, schema enforcement, and version history.

This is the source of truth. Embeddings are derived from it, never the
other way round.
"""

from pathlib import Path

import pandas as pd
from deltalake import DeltaTable, write_deltalake

TABLE_PATH = Path("data/delta/jobs")


def write_jobs(jobs: list[dict], path: Path = TABLE_PATH) -> int:
    """Append clean postings to the table. Creates it on first write.
    Returns the table version after the write."""
    df = pd.DataFrame(jobs)
    # mode="append" adds a new version. It never rewrites existing rows.
    # If the columns don't match the existing table, deltalake raises.
    write_deltalake(str(path), df, mode="append")
    return DeltaTable(str(path)).version()


def read_jobs(path: Path = TABLE_PATH, version: int | None = None) -> pd.DataFrame:
    """Read the table. Pass a version to read it as it was at that point."""
    table = DeltaTable(str(path), version=version)
    return table.to_pandas()


if __name__ == "__main__":
    sample = [
        {"id": "t-1", "title": "Data Engineer", "company": "Elm", "location": "Riyadh, SA",
         "description": "Build pipelines.", "skills": ["python", "sql"],
         "posted_at": "2026-09-01", "url": "https://example.com/1"},
        {"id": "t-2", "title": "ML Engineer", "company": "Mozn", "location": "Remote",
         "description": "Ship models.", "skills": ["python", "pytorch"],
         "posted_at": "2026-09-02", "url": "https://example.com/2"},
    ]
    demo_path = Path("data/delta/_demo")

    version = write_jobs(sample, demo_path)
    print(f"wrote {len(sample)} rows, table is now version {version}")

    df = read_jobs(demo_path)
    print(f"read back {len(df)} rows:")
    print(df[["id", "title", "company", "posted_at"]].to_string(index=False))

    print("\nfiles on disk:")
    for p in sorted(demo_path.rglob("*")):
        if p.is_file():
            print("  ", p.relative_to(demo_path))

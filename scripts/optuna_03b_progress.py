"""Print 03b Optuna progress from SQLite (run anytime; no notebook needed)."""
import sqlite3
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "outputs" / "stage03" / "03b"
TARGET = 2000


def study_progress(db: Path) -> tuple[int, float | None, dict[str, int]]:
    con = sqlite3.connect(db)
    try:
        state_rows = con.execute("SELECT state, COUNT(*) FROM trials GROUP BY state").fetchall()
        states = {str(s): int(c) for s, c in state_rows}
        complete = states.get("COMPLETE", 0)
        row = con.execute(
            """
            SELECT MAX(tv.value) FROM trial_values tv
            JOIN trials t ON t.trial_id = tv.trial_id
            WHERE t.state = 'COMPLETE'
            """
        ).fetchone()
        best = float(row[0]) if row and row[0] is not None else None
        return complete, best, states
    finally:
        con.close()


def main() -> None:
    if not ROOT.is_dir():
        print(f"No 03b folder: {ROOT}")
        return

    any_db = False
    for bucket in sorted(p for p in ROOT.iterdir() if p.is_dir()):
        dbs = sorted(bucket.glob("optuna_*.sqlite3"))
        configs = sorted(bucket.glob("best_configs_*.json"))
        if not dbs and not configs:
            continue
        print("=" * 64)
        print(bucket.name)
        for db in dbs:
            any_db = True
            complete, best, states = study_progress(db)
            remaining = max(0, TARGET - complete)
            pct = 100.0 * complete / TARGET
            mtime = datetime.fromtimestamp(db.stat().st_mtime)
            print(f"  {db.name}")
            print(f"    progress: {complete}/{TARGET} ({pct:.1f}%)  |  {remaining} trials left")
            print(f"    best objective: {best}")
            print(f"    last DB write: {mtime}")
            if states.get("RUNNING"):
                print(f"    RUNNING trials: {states['RUNNING']} (kernel may still be active)")
        if configs:
            print("  written configs:")
            for f in configs:
                print(f"    {f.name} @ {datetime.fromtimestamp(f.stat().st_mtime)}")

    if not any_db:
        print("No optuna_*.sqlite3 files yet under outputs/stage03/03b/")
        print("Optuna creates the DB when the first gallery JOB starts (after 'JOB: bristol_gorillas').")


if __name__ == "__main__":
    main()

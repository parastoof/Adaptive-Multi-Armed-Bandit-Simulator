import sqlite3
from storage.database import DB_PATH


def test_database_integrity():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON")

    print(f"\nDatabase: {DB_PATH}")

    # 1) at least one experiment
    n_exp = conn.execute(
        "SELECT COUNT(*) FROM experiments"
    ).fetchone()[0]
    print(f"Experiments: {n_exp}")
    assert n_exp >= 1, "No experiments found in database"

    # 2) each experiment has exactly 3*2*50 = 300 runs
    rows = conn.execute(
        "SELECT experiment_id, COUNT(*) FROM runs GROUP BY experiment_id"
    ).fetchall()
    for exp_id, count in rows:
        print(f"  {exp_id}: {count} runs")
        assert count == 300, f"Expected 300 runs, got {count} for {exp_id}"

    # 3) each experiment has exactly 300,000 steps
    rows = conn.execute(
        """
        SELECT r.experiment_id, COUNT(*)
        FROM runs r
        JOIN step_results s ON r.run_id = s.run_id
        GROUP BY r.experiment_id
        """
    ).fetchall()
    for exp_id, count in rows:
        print(f"  {exp_id}: {count} step records")
        assert count == 300_000, f"Expected 300,000 steps, got {count}"

    # 4) no FK violations
    violations = conn.execute("PRAGMA foreign_key_check").fetchall()
    assert violations == [], f"FK violations: {violations}"

    # 5) no duplicate step records
    dups = conn.execute(
        """
        SELECT COUNT(*) FROM (
            SELECT run_id, step FROM step_results
            GROUP BY run_id, step HAVING COUNT(*) > 1
        )
        """
    ).fetchone()[0]
    assert dups == 0, f"{dups} duplicate (run_id, step) pairs"

    # 6) rewards ∈ {0, 1}
    bad = conn.execute(
        "SELECT COUNT(*) FROM step_results WHERE reward NOT IN (0, 1)"
    ).fetchone()[0]
    assert bad == 0, f"{bad} rows with reward not in {{0,1}}"

    # 7) optimal_action ∈ {0, 1}
    bad = conn.execute(
        "SELECT COUNT(*) FROM step_results WHERE optimal_action NOT IN (0, 1)"
    ).fetchone()[0]
    assert bad == 0, f"{bad} rows with optimal_action not in {{0,1}}"

    conn.close()
    print("\nPASS: database integrity check")


if __name__ == "__main__":
    test_database_integrity()
import sqlite3

from storage.database import get_connection, save_experiment


def test_duplicate_experiment_rejected():
    # Use the most recent experiment in the DB (works for any repo clone)
    with get_connection() as conn:
        row = conn.execute(
            "SELECT experiment_id, master_seed, n_runs, n_steps "
            "FROM experiments ORDER BY rowid DESC LIMIT 1"
        ).fetchone()

    if row is None:
        raise RuntimeError(
            "No experiments in database. Run `python main.py` first."
        )

    exp_id, master_seed, n_runs, n_steps = row
    print(f"Using existing experiment: {exp_id}")

    with get_connection() as conn:
        before = conn.execute(
            "SELECT * FROM experiments WHERE experiment_id = ?",
            (exp_id,),
        ).fetchone()
        count_before = conn.execute(
            "SELECT COUNT(*) FROM experiments"
        ).fetchone()[0]

    print("Attempting to insert duplicate...")
    try:
        save_experiment(exp_id, master_seed, n_runs, n_steps)
        raise AssertionError("Duplicate experiment was accepted!")
    except sqlite3.IntegrityError as e:
        print(f"PASS: duplicate rejected ({e})")

    with get_connection() as conn:
        after = conn.execute(
            "SELECT * FROM experiments WHERE experiment_id = ?",
            (exp_id,),
        ).fetchone()
        count_after = conn.execute(
            "SELECT COUNT(*) FROM experiments"
        ).fetchone()[0]

    assert before == after, "Existing experiment was modified!"
    assert count_before == count_after, "Experiment count changed!"
    print("PASS: existing experiment preserved")


if __name__ == "__main__":
    test_duplicate_experiment_rejected()
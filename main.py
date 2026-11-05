import argparse
import sqlite3
from datetime import datetime

from experiments.experiment_runner import (
    MASTER_SEED, ENVIRONMENTS, AGENTS,
    N_RUNS, N_STEPS,
    create_seed_schedule, run_experiments,
)
from storage.database import (
    DB_PATH, get_connection, create_tables,
    save_experiment, save_all_runs, save_step_results,
)
from visualization.plots import generate_plots


def generate_experiment_id() -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return f"bandit_exp_{ts}"


def get_latest_experiment_id() -> str:
    if not DB_PATH.exists():
        raise FileNotFoundError(
            "Database not found. Run 'python main.py' first."
        )

    try:
        with get_connection() as conn:
            row = conn.execute(
                """
                SELECT experiment_id FROM experiments
                ORDER BY created_at DESC, rowid DESC LIMIT 1
                """
            ).fetchone()
    except sqlite3.OperationalError as e:
        raise RuntimeError(
            "Database tables are missing. Run 'python main.py' first."
        ) from e

    if row is None:
        raise RuntimeError("No experiments found in the database.")
    return row[0]


def run_new_experiment() -> str:
    experiment_id = generate_experiment_id()

    print("\nStarting new experiment...")
    print(f"Experiment ID: {experiment_id}")

    # 1) Simulate
    results = run_experiments()

    # 2) Rebuild deterministic seed schedule (matches runner exactly)
    env_seeds, agent_seeds = create_seed_schedule(
        environments=ENVIRONMENTS,
        agents=AGENTS,
        n_runs=N_RUNS,
        master_seed=MASTER_SEED,
    )

    # 3) Persist (order matters for FK)
    create_tables()

    try:
        save_experiment(
            experiment_id=experiment_id,
            master_seed=MASTER_SEED,
            n_runs=N_RUNS,
            n_steps=N_STEPS,
        )

        run_ids = save_all_runs(
            experiment_id=experiment_id,
            env_seeds=env_seeds,
            agent_seeds=agent_seeds,
            environments=ENVIRONMENTS,
            agents=AGENTS,
            n_runs=N_RUNS,
        )

        total_saved = save_step_results(results, run_ids)

    except Exception:
        # Best-effort cleanup so we don't leave a half-written experiment
        with get_connection() as conn:
            conn.execute(
                "DELETE FROM step_results WHERE run_id IN "
                "(SELECT run_id FROM runs WHERE experiment_id = ?)",
                (experiment_id,),
            )
            conn.execute(
                "DELETE FROM runs WHERE experiment_id = ?",
                (experiment_id,),
            )
            conn.execute(
                "DELETE FROM experiments WHERE experiment_id = ?",
                (experiment_id,),
            )
        raise

    # 4) Plots from DB
    generate_plots(experiment_id)

    print("\nExperiment completed successfully.")
    print(f"Experiment ID: {experiment_id}")
    print(f"Total runs: {len(run_ids):,}")
    print(f"Total step records: {total_saved:,}")

    return experiment_id


def main():
    parser = argparse.ArgumentParser(
        description="Adaptive Multi-Armed Bandit Simulator"
    )
    parser.add_argument(
        "mode", nargs="?", choices=["run", "plots"], default="run",
    )
    parser.add_argument("experiment_id", nargs="?")

    args = parser.parse_args()

    if args.mode == "run":
        if args.experiment_id:
            parser.error("experiment_id can only be used with plots mode.")
        run_new_experiment()

    elif args.mode == "plots":
        experiment_id = args.experiment_id or get_latest_experiment_id()
        print(f"Generating plots for: {experiment_id}")
        try:
            generate_plots(experiment_id)
        except RuntimeError as e:
            parser.error(str(e))


if __name__ == "__main__":
    main()
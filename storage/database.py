import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

DB_PATH = Path(__file__).resolve().parents[1] / "bandit_results.db"


def get_connection():
    connection = sqlite3.connect(DB_PATH)
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def create_tables(verbose: bool = True):
    with get_connection() as conn:
        cur = conn.cursor()

        cur.execute("""
            CREATE TABLE IF NOT EXISTS experiments (
                experiment_id TEXT PRIMARY KEY,
                master_seed   INTEGER NOT NULL,
                n_runs        INTEGER NOT NULL,
                n_steps       INTEGER NOT NULL,
                created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS runs (
                run_id        INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_id TEXT NOT NULL,
                environment   TEXT NOT NULL,
                agent         TEXT NOT NULL,
                run_index     INTEGER NOT NULL,
                env_seed      INTEGER NOT NULL,
                agent_seed    INTEGER NOT NULL,
                epsilon       REAL,
                alpha         REAL,
                FOREIGN KEY (experiment_id)
                    REFERENCES experiments(experiment_id),
                UNIQUE (experiment_id, environment, agent, run_index)
            )
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS step_results (
                run_id                   INTEGER NOT NULL,
                step                     INTEGER NOT NULL,
                action                   INTEGER NOT NULL,
                reward                   INTEGER NOT NULL
                    CHECK (reward IN (0, 1)),
                cumulative_reward        INTEGER NOT NULL,
                optimal_action           INTEGER NOT NULL
                    CHECK (optimal_action IN (0, 1)),
                expected_regret          REAL NOT NULL,
                cumulative_expected_regret REAL NOT NULL,
                PRIMARY KEY (run_id, step),
                FOREIGN KEY (run_id) REFERENCES runs(run_id)
            )
        """)

        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_runs_env_agent
                ON runs(environment, agent)
        """)
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_steps_step
                ON step_results(step)
        """)

    if verbose:
        print("Database tables created (or already exist).")


def save_experiment(experiment_id, master_seed, n_runs, n_steps):
    """Insert a new experiment. Raises sqlite3.IntegrityError on duplicate."""
    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO experiments (
                experiment_id, master_seed, n_runs, n_steps
            ) VALUES (?, ?, ?, ?)
            """,
            (experiment_id, master_seed, n_runs, n_steps),
        )


def save_all_runs(
    experiment_id, env_seeds, agent_seeds,
    environments, agents, n_runs,
):
    """
    Insert all run-metadata rows for one experiment.
    Returns dict[(env, agent, run_index)] -> run_id.
    """
    run_ids = {}

    with get_connection() as conn:
        cur = conn.cursor()

        for env_name in environments:
            for agent_name in agents:
                epsilon = 0.1 if agent_name != "random" else None
                alpha = 0.1 if agent_name == "constant_step" else None

                for run_idx in range(n_runs):
                    env_seed = env_seeds[(env_name, run_idx)]
                    agent_seed = agent_seeds[
                        (env_name, agent_name, run_idx)
                    ]

                    cur.execute(
                        """
                        INSERT INTO runs (
                            experiment_id, environment, agent, run_index,
                            env_seed, agent_seed, epsilon, alpha
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            experiment_id, env_name, agent_name, run_idx,
                            env_seed, agent_seed, epsilon, alpha,
                        ),
                    )
                    run_ids[(env_name, agent_name, run_idx)] = cur.lastrowid

    return run_ids


def save_step_results(results, run_ids):
    """Bulk-insert every step-level record for all (env, agent)."""
    insert_sql = """
        INSERT INTO step_results (
            run_id, step, action, reward, cumulative_reward,
            optimal_action, expected_regret, cumulative_expected_regret
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """

    total = 0

    with get_connection() as conn:
        cur = conn.cursor()

        for (env_name, agent_name), data in results.items():
            n_runs, n_steps = data["rewards"].shape

            for run_idx in range(n_runs):
                run_id = run_ids[(env_name, agent_name, run_idx)]

                cum_regrets = np.cumsum(
                    data["expected_regrets"][run_idx]
                )

                rows = [
                    (
                        run_id,
                        step + 1,
                        int(data["actions"][run_idx, step]),
                        int(data["rewards"][run_idx, step]),
                        int(data["cumulative_rewards"][run_idx, step]),
                        int(data["optimal_actions"][run_idx, step]),
                        float(data["expected_regrets"][run_idx, step]),
                        float(cum_regrets[step]),
                    )
                    for step in range(n_steps)
                ]

                cur.executemany(insert_sql, rows)
                total += len(rows)

    print(f"Saved {total:,} step records.")
    return total


# ----------------------------------------------------------------------
# Read helpers
# ----------------------------------------------------------------------
def get_average_rewards(experiment_id):
    query = """
        SELECT
            r.environment,
            r.agent,
            ROUND(AVG(s.reward), 5)          AS average_reward,
            ROUND(AVG(s.optimal_action), 5)  AS optimal_action_rate,
            COUNT(*)                         AS total_steps
        FROM runs r
        JOIN step_results s ON r.run_id = s.run_id
        WHERE r.experiment_id = ?
        GROUP BY r.environment, r.agent
        ORDER BY r.environment, average_reward DESC
    """
    with get_connection() as conn:
        return conn.execute(query, (experiment_id,)).fetchall()


def get_metric_curves(experiment_id) -> pd.DataFrame:
    query = """
        SELECT
            r.environment,
            r.agent,
            s.step,
            AVG(s.reward)                    AS average_reward_per_step,
            AVG(s.cumulative_reward)         AS average_cumulative_reward,
            AVG(s.optimal_action)            AS optimal_action_rate,
            AVG(s.cumulative_expected_regret)
                AS average_cumulative_expected_regret
        FROM runs r
        JOIN step_results s ON r.run_id = s.run_id
        WHERE r.experiment_id = ?
        GROUP BY r.environment, r.agent, s.step
        ORDER BY r.environment, r.agent, s.step
    """
    with get_connection() as conn:
        return pd.read_sql_query(query, conn, params=(experiment_id,))
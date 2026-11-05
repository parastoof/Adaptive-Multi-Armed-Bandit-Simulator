"""
Extract all numbers needed for README/REPORT from the SQLite database.
Run after `python main.py`.
"""
from storage.database import get_connection


def fetch_latest_experiment_id():
    with get_connection() as conn:
        row = conn.execute(
            "SELECT experiment_id FROM experiments "
            "ORDER BY created_at DESC, rowid DESC LIMIT 1"
        ).fetchone()
    return row[0]


def main():
    exp_id = fetch_latest_experiment_id()
    print(f"Experiment: {exp_id}\n")

    with get_connection() as conn:
        # ---- 1) Average reward + optimal action rate (overall) ----
        print("=" * 70)
        print("TABLE 1: Average Reward and Optimal Action Rate (overall)")
        print("=" * 70)
        rows = conn.execute("""
            SELECT
                r.environment,
                r.agent,
                AVG(s.reward)         AS avg_reward,
                AVG(s.optimal_action) AS opt_rate
            FROM runs r
            JOIN step_results s ON r.run_id = s.run_id
            WHERE r.experiment_id = ?
            GROUP BY r.environment, r.agent
            ORDER BY r.environment, avg_reward DESC
        """, (exp_id,)).fetchall()

        print(f"{'Environment':<16} {'Agent':<16} {'Avg Reward':>12} {'Opt Rate':>12}")
        print("-" * 70)
        for env, agent, avg_r, opt_r in rows:
            print(f"{env:<16} {agent:<16} {avg_r:>12.5f} {opt_r:>12.5f}")

        # ---- 2) Final cumulative expected regret (at step 1000) ----
        print("\n" + "=" * 70)
        print("TABLE 2: Final Cumulative Expected Regret (at step 1000)")
        print("=" * 70)
        rows = conn.execute("""
            SELECT
                r.environment,
                r.agent,
                AVG(s.cumulative_expected_regret) AS final_regret
            FROM runs r
            JOIN step_results s ON r.run_id = s.run_id
            WHERE r.experiment_id = ? AND s.step = 1000
            GROUP BY r.environment, r.agent
            ORDER BY r.environment, final_regret ASC
        """, (exp_id,)).fetchall()

        print(f"{'Environment':<16} {'Agent':<16} {'Final Regret':>15}")
        print("-" * 70)
        for env, agent, regret in rows:
            print(f"{env:<16} {agent:<16} {regret:>15.2f}")

        # ---- 3) Regret at step 500 (before change) and step 1000 (after) ----
        print("\n" + "=" * 70)
        print("TABLE 3: Cumulative Regret at Step 500 vs Step 1000 (non-stationary only)")
        print("=" * 70)
        rows = conn.execute("""
            SELECT
                r.agent,
                s.step,
                AVG(s.cumulative_expected_regret) AS avg_regret
            FROM runs r
            JOIN step_results s ON r.run_id = s.run_id
            WHERE r.experiment_id = ?
              AND r.environment = 'non_stationary'
              AND s.step IN (500, 1000)
            GROUP BY r.agent, s.step
            ORDER BY r.agent, s.step
        """, (exp_id,)).fetchall()

        print(f"{'Agent':<16} {'Step':>6} {'Cum Regret':>14}")
        print("-" * 70)
        for agent, step, regret in rows:
            print(f"{agent:<16} {step:>6} {regret:>14.2f}")

        # ---- 4) Recovery: optimal action rate around step 501 ----
        print("\n" + "=" * 70)
        print("TABLE 4: Optimal Action Rate around Change (non-stationary)")
        print("=" * 70)
        rows = conn.execute("""
            SELECT
                r.agent,
                s.step,
                AVG(s.optimal_action) AS opt_rate
            FROM runs r
            JOIN step_results s ON r.run_id = s.run_id
            WHERE r.experiment_id = ?
              AND r.environment = 'non_stationary'
              AND s.step IN (500, 510, 520, 550, 600, 700, 800, 900, 1000)
            GROUP BY r.agent, s.step
            ORDER BY r.agent, s.step
        """, (exp_id,)).fetchall()

        # pivot for readability
        from collections import defaultdict
        pivot = defaultdict(dict)
        for agent, step, rate in rows:
            pivot[agent][step] = rate

        steps = [500, 510, 520, 550, 600, 700, 800, 900, 1000]
        header = f"{'Agent':<16}" + "".join(f"{s:>8}" for s in steps)
        print(header)
        print("-" * len(header))
        for agent in ["random", "sample_average", "constant_step"]:
            line = f"{agent:<16}"
            for s in steps:
                v = pivot[agent].get(s, float("nan"))
                line += f"{v:>8.3f}"
            print(line)

        # ---- 5) Stationary: optimal action rate around step 100, 500, 1000 ----
        print("\n" + "=" * 70)
        print("TABLE 5: Optimal Action Rate over Time (stationary)")
        print("=" * 70)
        rows = conn.execute("""
            SELECT
                r.agent,
                s.step,
                AVG(s.optimal_action) AS opt_rate
            FROM runs r
            JOIN step_results s ON r.run_id = s.run_id
            WHERE r.experiment_id = ?
              AND r.environment = 'stationary'
              AND s.step IN (10, 50, 100, 200, 500, 1000)
            GROUP BY r.agent, s.step
            ORDER BY r.agent, s.step
        """, (exp_id,)).fetchall()

        pivot = defaultdict(dict)
        for agent, step, rate in rows:
            pivot[agent][step] = rate

        steps = [10, 50, 100, 200, 500, 1000]
        header = f"{'Agent':<16}" + "".join(f"{s:>8}" for s in steps)
        print(header)
        print("-" * len(header))
        for agent in ["random", "sample_average", "constant_step"]:
            line = f"{agent:<16}"
            for s in steps:
                v = pivot[agent].get(s, float("nan"))
                line += f"{v:>8.3f}"
            print(line)

    print("\n" + "=" * 70)
    print(f"Experiment ID for README/REPORT: {exp_id}")
    print("=" * 70)


if __name__ == "__main__":
    main()
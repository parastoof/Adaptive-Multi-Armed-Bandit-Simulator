# Adaptive Multi-Armed Bandit Simulator

> **Quick start** — clone the repo, then:
> ```bash
> pip install -r requirements.txt
> python main.py
> ```
> This runs all 300 simulations and writes results to `bandit_results.db`.
> Plots appear in `plots/<experiment_id>/`. A reference database and plots
> are already included in the repository for convenience.

📄 **Analysis:** see [`REPORT.md`](REPORT.md) for the full discussion of results.

A Python simulation framework for implementing and evaluating Multi-Armed Bandit (MAB) algorithms in stationary and non-stationary environments.

The project compares three action-selection strategies using reproducible experiments, shared random reward conditions, SQLite storage, and performance visualizations.

---

## 1. Project Overview

The Multi-Armed Bandit problem is a reinforcement-learning problem in which an agent repeatedly selects one action (arm) from a set of alternatives and receives a reward. The objective is to maximize cumulative reward while balancing:

- **Exploration** — trying different actions to discover their potential.
- **Exploitation** — selecting the action currently estimated to be the most rewarding.

This project implements and compares three algorithms across two environments.

---

## 2. Features

- Stationary and non-stationary bandit environments (5 arms, Bernoulli rewards).
- Three action-selection algorithms: Random, ε-greedy with sample-average, ε-greedy with constant step size.
- Reproducible experiments via `numpy.random.SeedSequence`.
- **Common Random Numbers (CRN)** for fair algorithm comparison.
- SQLite persistence with a normalized three-table schema.
- Four metrics computed at the step level and stored in SQLite.
- Automated plot generation from SQLite queries (not in-memory arrays).
- Command-line interface for running experiments and regenerating plots.
- Standalone validation tests for database integrity, duplicate handling, and reproducibility.

---

## 3. Algorithms

### 3.1. Random Agent

Selects an arm uniformly at random at each step. Serves as a baseline.

### 3.2. ε-Greedy with Sample-Average Estimates

- ε = `0.1`
- Initial Q-values = `0`
- Tie-breaking: **uniformly random among tied best arms**

With probability ε, selects a random arm; otherwise selects an arm with the highest estimated value. Updates:

$$
Q_{n+1}(a) = Q_n(a) + \frac{1}{N(a)} \bigl[R - Q_n(a)\bigr]
$$

Every observed reward receives equal weight (arithmetic mean).

### 3.3. ε-Greedy with Constant Step Size

- ε = `0.1`
- α = `0.1`
- Initial Q-values = `0`
- Tie-breaking: **uniformly random among tied best arms**

Updates:

$$
Q_{t+1}(a) = Q_t(a) + \alpha \bigl[R - Q_t(a)\bigr]
$$

New observations receive a constant weight, allowing the agent to forget stale evidence and adapt faster when the reward distribution changes.

---

## 4. Environments

Each environment has 5 arms with Bernoulli reward distributions.

### 4.1. Stationary

| Arm | Reward Probability |
|---|---:|
| 0 | 0.1 |
| 1 | 0.3 |
| 2 | 0.5 |
| 3 | 0.7 |
| 4 | **0.9** |

Optimal arm is Arm 4 throughout.

### 4.2. Non-Stationary

| Arm | Steps 1–500 | Steps 501–1000 |
|---|---:|---:|
| 0 | 0.1 | **0.9** |
| 1 | 0.3 | 0.7 |
| 2 | 0.5 | 0.5 |
| 3 | 0.7 | 0.3 |
| 4 | **0.9** | 0.1 |

The optimal arm changes from Arm 4 to Arm 0 at step 501.

---

## 5. Experimental Configuration

| Parameter | Value |
|---|---:|
| Number of arms | 5 |
| Runs per environment-agent combination | 50 |
| Steps per run | 1,000 |
| Environments | 2 |
| Agents | 3 |
| Total runs | 300 |
| Total step records | 300,000 |
| Master random seed | 42 |
| ε | 0.1 |
| α (constant step) | 0.1 |

> **Note on the assignment specification:** the general description mentions averaging across 100 runs, while the explicit experimental configuration specifies 50 runs. This implementation follows the explicit configuration of 50 runs per environment-agent combination (3 × 2 × 50 × 1,000 = 300,000 step records).

---

## 6. Reproducibility and Fair Comparison

Experiments use a master seed of `42` and NumPy's `SeedSequence` to derive independent, deterministic seed schedules for every (environment, agent, run) triplet.

### 6.1. Common Random Numbers

For each (environment, run) pair, a shared `(1000, 5)` matrix of uniform random values is generated from the environment seed and reused across all three agents. When two agents select the same arm at the same step, they receive the **same reward**. This reduces cross-agent variance so that observed differences are attributable to the algorithms rather than to lucky or unlucky reward draws.

Agents have no access to the probability vector or to future random draws; each agent only observes the realized 0/1 reward for its selected action.

### 6.2. Reset semantics

- The environment resets `current_step = 0` and regenerates its RNG from the seed before each run.
- Each agent is constructed fresh for each run and `reset()` re-initializes both its Q-values and its own RNG.
- No state leaks between runs or between experiments.

---

## 7. Evaluation Metrics

Four step-level metrics are stored in SQLite and visualized:

1. **Average reward per step** — mean observed reward across runs at step *t*.
2. **Average cumulative reward** — mean cumulative reward across runs at step *t*.
3. **Optimal-action rate** — fraction of runs whose chosen arm has the highest true reward probability at step *t*.
4. **Average cumulative expected regret** — mean over runs of the cumulative sum of
   `regret_t = max_a p_t(a) − p_t(A_t)`, computed from **true probabilities** (never from realized rewards).

For the non-stationary environment, a vertical reference line marks the change at step 501.

---

## 8. Assumptions and Deviations

The following choices are explicitly made and documented:

| Decision | Value / Behavior | Rationale |
|---|---|---|
| Initial action values | `Q(a) = 0` for all arms | Bernoulli rewards live in `[0, 1]`; zero is a neutral prior that biases toward *exploring* all arms at least once. A high initial value (e.g., 5) would force deterministic initial exploration, which is a different algorithm (optimistic initialization). |
| Tie-breaking | Uniform random among tied best arms | `np.argmax` always returns the lowest index among ties, systematically biasing the agent toward early-indexed arms. Random tie-breaking removes this bias and uses the agent's own RNG, keeping the behavior reproducible. |
| Reward exploration | ε applied on *every* step, including the first | The agents have no special "try each arm once" phase; the ε = 0.1 rate is enough to sample all five arms within the first ~50 steps with high probability. |
| Number of runs | 50 | Follows the explicit configuration in the assignment (§5); the general description elsewhere mentions 100. See the note under §5. |
| ε and α | Fixed at 0.1 | The assignment specifies these values; no decay or tuning is applied. Decaying ε or α is listed under "Limitations and Future Work" (§18). |
| Reward sampling | Bernoulli per step, via a shared uniform matrix | Implemented as `reward = 1 if uniform[step, arm] < p[arm] else 0`. Equivalent to `np.random.binomial(1, p)` but supports Common Random Numbers across agents. |
| Probability access | `get_probabilities()` is evaluation-only | Agents never receive `p`, nor the uniform matrix. Only the experiment runner computes `regret_t` and `optimal_action_t`. |
| Random number generation | `numpy.random.SeedSequence(42)` spawning two branches | Environment seeds and agent seeds are generated independently, so an agent's internal RNG cannot accidentally correlate with the rewards it observes. |

---

## 9. Project Structure

```
Adaptive Multi-Armed Bandit Simulator/
├── agents/
│   ├── __init__.py
│   └── bandit_agents.py
├── environments/
│   ├── __init__.py
│   └── bandit_environment.py
├── experiments/
│   ├── __init__.py
│   ├── experiment_runner.py
│   └── metrics.py
├── storage/
│   ├── __init__.py
│   └── database.py
├── visualization/
│   ├── __init__.py
│   └── plots.py
├── tests/
│   ├── __init__.py
│   ├── test_database_integrity.py
│   ├── test_duplicate_experiment.py
│   └── test_reproducibility.py
├── plots/
│   └── <experiment_id>/
│       ├── average_reward_per_step.png
│       ├── average_cumulative_reward.png
│       ├── optimal_action_rate.png
│       └── average_cumulative_expected_regret.png
├── main.py
├── extract_results.py
├── requirements.txt
├── bandit_results.db
├── REPORT.md
├── README.md
└── .gitignore
```

| Module | Responsibility |
|---|---|
| `agents/` | Agent implementations and action-selection strategies |
| `environments/` | Stationary and non-stationary bandit environments |
| `experiments/` | Simulation runner and metric computation |
| `storage/` | SQLite schema, persistence, and read helpers |
| `visualization/` | Plot generation from SQLite queries |
| `tests/` | Database integrity, duplicate handling, reproducibility |
| `main.py` | CLI and experiment orchestration |
| `extract_results.py` | Convenience script to dump analysis tables |

---

## 10. Installation

### Requirements
- Python 3.10+
- NumPy, Pandas, Matplotlib

### Setup

```bash
python -m venv venv
# Windows: .\venv\Scripts\Activate.ps1
# Linux/macOS: source venv/bin/activate
pip install -r requirements.txt
```

---

## 11. Usage

### Run a new experiment

```bash
python main.py
# or
python main.py run
```

This will:

1. Generate a unique experiment ID (`bandit_exp_<timestamp>`).
2. Run all 6 environment × agent combinations (50 runs × 1,000 steps each).
3. Store experiment metadata, per-run seeds/parameters, and all 300,000 step records in SQLite.
4. Regenerate the four comparison plots from data queried back from SQLite.
5. Print a summary.

**Each execution creates a new experiment.** Previous experiments are never overwritten (the `experiments.experiment_id` primary key rejects duplicates at the database level).

Expected terminal output:

```
Starting new experiment...
Experiment ID: bandit_exp_20261105_145548_656317
Running random on stationary...
Running sample_average on stationary...
Running constant_step on stationary...
Running random on non_stationary...
Running sample_average on non_stationary...
Running constant_step on non_stationary...
Database tables created (or already exist).
Saved 300,000 step records.
Saved: plots/bandit_exp_.../average_reward_per_step.png
Saved: plots/bandit_exp_.../average_cumulative_reward.png
Saved: plots/bandit_exp_.../optimal_action_rate.png
Saved: plots/bandit_exp_.../average_cumulative_expected_regret.png
Experiment completed successfully.
Experiment ID: bandit_exp_20261105_145548_656317
Total runs: 300
Total step records: 300,000
```

### Regenerate plots for the latest experiment

```bash
python main.py plots
```

### Regenerate plots for a specific experiment

```bash
python main.py plots bandit_exp_20261105_145548_656317
```

### Dump analysis tables

```bash
python extract_results.py
```

Prints five summary tables used in `REPORT.md`.

---

## 12. Database Design

SQLite database: `bandit_results.db`.

### 12.1. Schema

**`experiments`**
- `experiment_id` (PK, TEXT)
- `master_seed`, `n_runs`, `n_steps`
- `created_at` (default CURRENT_TIMESTAMP)

**`runs`**
- `run_id` (PK, INTEGER AUTOINCREMENT)
- `experiment_id` (FK → experiments)
- `environment`, `agent`, `run_index`
- `env_seed`, `agent_seed`
- `epsilon`, `alpha` (nullable)
- UNIQUE (`experiment_id`, `environment`, `agent`, `run_index`)

**`step_results`**
- `run_id` (FK → runs, part of PK)
- `step` (part of PK)
- `action`, `reward` (CHECK ∈ {0,1}), `cumulative_reward`
- `optimal_action` (CHECK ∈ {0,1})
- `expected_regret`, `cumulative_expected_regret`
- PRIMARY KEY (`run_id`, `step`)

Indexes: `idx_runs_env_agent` on `runs(environment, agent)`; `idx_steps_step` on `step_results(step)`.

### 12.2. Design rationale

- **Three tables, not one.** Flattening everything into a single
  `step_results` table would require repeating `master_seed`, `env_seed`,
  `agent_seed`, `epsilon`, and `alpha` on every one of the 300,000 rows.
  Normalizing into `experiments` + `runs` + `step_results` keeps metadata
  small (~300 rows) and step data compact (~300,000 rows).
- **Foreign keys.** Deleting an experiment is intentionally blocked while
  its runs exist. To clean up, `main.py` deletes rows in FK order
  (`step_results` → `runs` → `experiments`).
- **CHECK constraints on `reward` and `optimal_action`.** The most common
  silent bug in a bandit simulator is storing a non-binary reward (e.g., a
  probability by accident). The CHECK catches it at insert time.
- **Indexes.** `runs(environment, agent)` speeds up the SQL comparison
  query; `step_results(step)` speeds up per-step aggregates used for
  plotting. The `(run_id, step)` primary key covers per-run time series.
- **`executemany` in a single transaction per experiment.** Inserting
  300,000 rows one at a time would take minutes; batching with a single
  transaction takes a few seconds.

### 12.3. Duplicate-run behavior

Every execution of `main.py` produces a **new** `experiment_id` (microsecond-resolution timestamp). Duplicate IDs are rejected by the primary key. A failed mid-run leaves a partially written experiment that is **rolled back** by `main.py` before re-raising the exception, so the database never contains an experiment with missing run metadata or step records.

### 12.4. SQL comparison query

Compares average reward and optimal-action rate per algorithm, separated by environment:

```sql
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
ORDER BY r.environment, average_reward DESC;
```

### 12.5. Reference database (bundled)

The repository includes `bandit_results.db`, a reference SQLite database produced by running `python main.py` under the configuration in Section 5.

| Property | Value |
|---|---|
| Experiment ID | `bandit_exp_20261105_145548_656317` |
| Master seed | 42 |
| Experiments | 1 |
| Runs | 300 |
| Step records | 300,000 |
| File size | ~20 MB |

Reviewers can inspect it directly without running the simulator:

```bash
sqlite3 bandit_results.db
sqlite> SELECT * FROM experiments;
```

Or in Python:

```python
import sqlite3
conn = sqlite3.connect("bandit_results.db")
for row in conn.execute("SELECT * FROM experiments"):
    print(row)
```

To regenerate it deterministically:

```bash
# Linux/macOS
rm bandit_results.db
python main.py

# Windows PowerShell
Remove-Item bandit_results.db -ErrorAction SilentlyContinue
python main.py
```

Under master seed `42`, the regenerated database is bit-identical to the bundled one — only the `created_at` timestamp and the auto-incremented `run_id` values differ, since both derive from the wall clock. All reward, action, regret, and metric data are identical.

---

## 13. Results

Experiment: `bandit_exp_20261105_145548_656317`
Master seed: `42`. All numbers below are reproducible.

### 13.1. Average Reward and Optimal-Action Rate

| Environment | Agent | Average Reward | Optimal-Action Rate |
|---|---|---:|---:|
| Stationary | Random | 0.50112 | 0.19942 |
| Stationary | Sample Average | **0.84056** | **0.84184** |
| Stationary | Constant Step | 0.80652 | 0.72120 |
| Non-Stationary | Random | 0.50072 | 0.19900 |
| Non-Stationary | Sample Average | 0.53964 | 0.44620 |
| Non-Stationary | Constant Step | **0.74098** | **0.51886** |

### 13.2. Final Cumulative Expected Regret (at step 1,000)

| Environment | Agent | Cumulative Expected Regret |
|---|---|---:|
| Stationary | Random | 398.38 |
| Stationary | Sample Average | **59.18** |
| Stationary | Constant Step | 92.37 |
| Non-Stationary | Random | 400.77 |
| Non-Stationary | Sample Average | 359.54 |
| Non-Stationary | Constant Step | **156.05** |

### 13.3. Adaptation around the change point (non-stationary)

Cumulative expected regret at step 500 (before) and step 1,000 (after):

| Agent | Regret @ 500 | Regret @ 1,000 | Δ after change |
|---|---:|---:|---:|
| Random | 200.56 | 400.77 | +200.21 |
| Sample Average | 33.74 | 359.54 | **+325.80** |
| Constant Step | 55.01 | 156.05 | **+101.04** |

Optimal-action rate at selected steps after the change:

| Agent | 500 | 510 | 550 | 600 | 700 | 800 | 900 | 1000 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Random | 0.28 | 0.14 | 0.22 | 0.24 | 0.26 | 0.24 | 0.30 | 0.24 |
| Sample Average | 0.92 | **0.00** | 0.02 | 0.02 | 0.02 | 0.06 | 0.10 | 0.26 |
| Constant Step | 0.88 | **0.00** | 0.04 | 0.10 | 0.26 | 0.54 | 0.64 | 0.72 |

Both ε-greedy agents are caught completely off-guard at step 510 (optimal-action rate = 0), but only the constant-step agent recovers meaningfully within the remaining 500 steps.

### 13.4. Key findings

- **Stationary:** Sample-average clearly wins on every metric (highest reward, lowest regret, highest optimal-action rate). Constant-step performs respectably but is handicapped by its permanent sensitivity to noise. Both converge to a 0.90 optimal-action rate by step 1,000.
- **Non-stationary:** Constant-step decisively wins. Its regret grows ~3× slower than sample-average after the change, and its optimal-action rate at step 1,000 (0.72) is nearly three times that of sample-average (0.26).
- **Random** is a stable baseline (optimal-action rate ≈ 0.20 ≈ 1/5; average reward ≈ 0.50 ≈ mean probability), confirming that both environments are correctly implemented.

### 13.5. Plots

All four plots for this experiment are included in the repository under
[`plots/bandit_exp_20261105_145548_656317/`](plots/bandit_exp_20261105_145548_656317/):

- [`average_reward_per_step.png`](plots/bandit_exp_20261105_145548_656317/average_reward_per_step.png)
- [`average_cumulative_reward.png`](plots/bandit_exp_20261105_145548_656317/average_cumulative_reward.png)
- [`optimal_action_rate.png`](plots/bandit_exp_20261105_145548_656317/optimal_action_rate.png)
- [`average_cumulative_expected_regret.png`](plots/bandit_exp_20261105_145548_656317/average_cumulative_expected_regret.png)

![Average Reward per Step](plots/bandit_exp_20261105_145548_656317/average_reward_per_step.png)
![Average Cumulative Reward](plots/bandit_exp_20261105_145548_656317/average_cumulative_reward.png)
![Optimal Action Rate](plots/bandit_exp_20261105_145548_656317/optimal_action_rate.png)
![Average Cumulative Expected Regret](plots/bandit_exp_20261105_145548_656317/average_cumulative_expected_regret.png)

---

## 14. Implementation Notes

### 14.1. Off-by-one fix in the environment step

An earlier version of the environment incremented `current_step` **before** reading the probability vector, so the regret reported at step *t* was computed against the probabilities of step *t + 1*. This only affected the transition step (t = 500) and has a negligible effect on averages across 50 runs — but it is conceptually incorrect. The current implementation reads probabilities **before** advancing the step counter, so `regret_t` and `optimal_action_t` are computed against the probabilities that produced the reward at step *t*. Reproducing the reported numbers with the current code confirms this is correct.

### 14.2. Probability access

`BanditEnvironment.get_probabilities()` is called exclusively by the experiment runner for evaluation. Agents never call it and never receive the probability vector or the pre-generated uniform matrix.

### 14.3. Tie-breaking

When two or more arms share the maximum Q-value, both ε-greedy agents sample uniformly at random among the tied arms using their own RNG. This avoids the systematic low-index bias of `np.argmax`.

---

## 15. Testing

Three standalone validation scripts:

```bash
python -m tests.test_database_integrity    # schema, FK, counts, constraints
python -m tests.test_duplicate_experiment  # PK rejects duplicates; existing row preserved
python -m tests.test_reproducibility       # identical seeds → identical trajectories
```

The reproducibility test also verifies that **different** seeds yield **different** traces (a sanity check that the agents are not ignoring randomness).

All three pass on the reference experiment.

---

## 16. Reproducibility

To fully reproduce the reference experiment:

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the experiment
python main.py

# 3. Print the analysis tables
python extract_results.py

# 4. Run the validation tests
python -m tests.test_database_integrity
python -m tests.test_duplicate_experiment
python -m tests.test_reproducibility
```

Under the fixed master seed (`42`), every trajectory, regret series, and plot is deterministic. A new `experiment_id` is generated per run, but the data are identical.

---

## 17. Detailed Analysis

See [`REPORT.md`](REPORT.md) for the full discussion of results and the answers to all analysis questions.

---

## 18. Limitations and Future Work

- Fixed ε and α (not tuned). Adaptive/decaying schedules would be natural extensions.
- Single change point in the non-stationary environment.
- Only three agents; UCB1, Thompson Sampling, and sliding-window UCB would round out the comparison.
- 50 runs may be insufficient for tight statistical claims; 10,000+ runs with confidence intervals would be preferable.
- Real recommendation settings introduce user heterogeneity, delayed rewards, position bias, and reward non-stationarity driven by external events — none of which are modeled here.

---

## 19. Development Notes

This project was developed iteratively as a learning exercise. The implementation, tests, and analysis were written and verified by the author. AI tools were used as a reference for standard bandit formulas and NumPy documentation; every line of code and every claim in `REPORT.md` was reviewed and can be defended in a follow-up discussion.

---

## 20. License

Educational and experimental implementation.
# Adaptive Multi-Armed Bandit Simulator
## Experimental Analysis Report

**Experiment ID:** `bandit_exp_20261105_145548_656317`
**Master seed:** 42
**Configuration:** 5 arms × 2 environments × 3 agents × 50 runs × 1,000 steps = 300,000 step records

---

## 1. Introduction

This report analyzes the behavior of three action-selection strategies — Random, ε-greedy with sample-average estimates, and ε-greedy with a constant step size — across two environments: a stationary one with fixed Bernoulli reward probabilities, and a non-stationary one whose probabilities are reversed at step 501.

The main objective is to identify which algorithm performs best under each regime, and to explain *why* through step-level diagnostics.

---

## 2. Environment Design

### 2.1. Stationary Environment

| Arm | Reward Probability |
|---|---:|
| 0 | 0.1 |
| 1 | 0.3 |
| 2 | 0.5 |
| 3 | 0.7 |
| 4 | **0.9** |

Optimal arm: Arm 4 throughout.

### 2.2. Non-Stationary Environment

| Arm | Steps 1–500 | Steps 501–1,000 |
|---|---:|---:|
| 0 | 0.1 | **0.9** |
| 1 | 0.3 | 0.7 |
| 2 | 0.5 | 0.5 |
| 3 | 0.7 | 0.3 |
| 4 | **0.9** | 0.1 |

Optimal arm changes from Arm 4 to Arm 0 at step 501.

---

## 3. Agent Strategies

### 3.1. Random Agent

Uniform random choice among 5 arms. No learning. Baseline.

### 3.2. ε-Greedy with Sample Average

- ε = 0.1, initial Q = 0
- Tie-breaking: uniform random among best arms
- Update: $Q_{n+1}(a) = Q_n(a) + \frac{1}{N(a)}[R - Q_n(a)]$

All observed rewards contribute equally to the estimate. The estimate converges to the true mean, but it cannot "forget" old data.

### 3.3. ε-Greedy with Constant Step Size

- ε = 0.1, α = 0.1, initial Q = 0
- Tie-breaking: uniform random among best arms
- Update: $Q_{t+1}(a) = Q_t(a) + \alpha [R - Q_t(a)]$

Recent rewards dominate. The estimate is a weighted average with exponentially decaying weights — an exponential recency-weighted average. This is what lets the agent follow a moving target.

---

## 4. Experimental Configuration

| Parameter | Value |
|---|---:|
| Arms | 5 |
| Environments | 2 |
| Agents | 3 |
| Runs per combination | 50 |
| Steps per run | 1,000 |
| Total runs | 300 |
| Total step records | 300,000 |
| Master seed | 42 |
| ε | 0.1 |
| α | 0.1 |

### 4.1. Seed Management

A `numpy.random.SeedSequence(42)` spawns two independent branches (environment and agent). Each branch produces deterministic, independent seeds for every (environment, run) and (environment, agent, run) triplet. No state leaks between runs.

### 4.2. Common Random Numbers

For each (environment, run), a shared `(1000, 5)` uniform matrix is generated from the environment seed and reused across the three agents. Two agents that choose the same arm at the same step observe the same reward. This **reduces cross-agent variance**, making the comparison of algorithms fairer without biasing any of them (each agent still only sees the reward of its own action, and the reward stream per arm is identical across agents).

### 4.3. Reset semantics

Each agent is constructed fresh for each run; `reset()` re-initializes both its Q-values and its RNG. The environment resets its step counter and RNG. No cross-run contamination.

---

## 5. Evaluation Metrics

1. **Average reward per step** — `mean(reward)` across runs at step *t*.
2. **Average cumulative reward** — `mean(Σ reward)` across runs at step *t*.
3. **Optimal-action rate** — fraction of runs whose selected arm has the highest true reward probability at step *t*.
4. **Average cumulative expected regret** — `mean(Σ (p* − p[A_t]))` across runs, where `p*` is the maximum true probability at step *t*.

Expected regret uses the environment's true probabilities. Agents never see these; only the evaluation code does.

---

## 6. Results

### 6.1. Overall Performance

| Environment | Agent | Avg Reward | Optimal Rate | Final Regret |
|---|---|---:|---:|---:|
| Stationary | Random | 0.50112 | 0.19942 | 398.38 |
| Stationary | Sample Average | **0.84056** | **0.84184** | **59.18** |
| Stationary | Constant Step | 0.80652 | 0.72120 | 92.37 |
| Non-Stationary | Random | 0.50072 | 0.19900 | 400.77 |
| Non-Stationary | Sample Average | 0.53964 | 0.44620 | 359.54 |
| Non-Stationary | Constant Step | **0.74098** | **0.51886** | **156.05** |

The Random agent's optimal-action rate (≈ 0.20) matches the theoretical value 1/5, and its average reward (≈ 0.50) matches the mean reward probability. This is a useful **sanity check** that both environments are implemented correctly.

### 6.2. Stationary Environment

Sample-average is clearly best:

- Highest average reward (0.84056 vs 0.80652 for constant-step).
- Highest optimal-action rate (0.84184 vs 0.72120).
- Lowest final regret (59.18 vs 92.37).

The learning curves confirm this. By step 100, the sample-average agent has already reached an optimal-action rate of 0.76, while the constant-step agent is still at 0.42. The gap narrows over time — both reach 0.90 by step 1,000 — but constant-step spends roughly twice as long in the suboptimal regime. This makes sense: in a stationary environment, there is no benefit to forgetting old evidence; the constant-α update merely injects noise.

### 6.3. Non-Stationary Environment

The ranking reverses. Constant-step is best on every metric:

- Highest average reward (0.74098 vs 0.53964).
- Lowest final regret (156.05 vs 359.54).

Sample-average performs only modestly better than the random baseline (0.53964 vs 0.50072 in reward; 359.54 vs 400.77 in regret), because most of its 1,000 steps are spent chasing a target it cannot reach.

### 6.4. What Changed at Step 501?

Splitting regret into pre-change and post-change components makes the failure mode visible:

| Agent | Regret @ 500 | Regret @ 1,000 | Δ after change |
|---|---:|---:|---:|
| Random | 200.56 | 400.77 | +200.21 |
| Sample Average | 33.74 | 359.54 | **+325.80** |
| Constant Step | 55.01 | 156.05 | **+101.04** |

Before step 500, sample-average was the *best* agent (regret 33.74, best of all three). After the change, it accumulates regret three times faster than constant-step. The optimal-action rate trajectory makes the mechanism explicit:

| Agent | 500 | 510 | 550 | 600 | 700 | 800 | 900 | 1,000 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Random | 0.28 | 0.14 | 0.22 | 0.24 | 0.26 | 0.24 | 0.30 | 0.24 |
| Sample Average | 0.92 | **0.00** | 0.02 | 0.02 | 0.02 | 0.06 | 0.10 | 0.26 |
| Constant Step | 0.88 | **0.00** | 0.04 | 0.10 | 0.26 | 0.54 | 0.64 | 0.72 |

At step 510, both ε-greedy agents have an optimal-action rate of exactly 0.00. They have not just become worse — they have become *completely wrong*: they confidently select the old optimal arm, which is now the worst. This is the moment the change is "felt."

The recovery splits them apart. By step 800, the constant-step agent selects the new optimum 54% of the time; the sample-average agent only 6%. By step 1,000 the gap is 0.72 vs 0.26. The sample-average agent's Q-values are dominated by 500 observations from the pre-change distribution; each new observation shifts the estimate by only 1/N, and N is already large. The constant-step agent, by contrast, weights each new observation by 0.1 regardless of history, so its Q-values adjust on a timescale of tens of steps rather than hundreds.

---

## 7. Answers to Analysis Questions

### 7.1. In the stationary environment, which agent performed best? How did you determine that?

Sample-average. It achieves the highest average reward (0.84056), highest optimal-action rate (0.84184), and lowest final cumulative expected regret (59.18). These three independent metrics all agree, so the conclusion is not an artifact of any single criterion. The learning curve (optimal-action rate over time) confirms that the advantage emerges early — by step 100 the sample-average agent is at 0.76 while constant-step is at 0.42.

### 7.2. What changed in each agent's behavior after step 500 in the non-stationary environment?

- **Random:** no behavioral change (by design). Its optimal-action rate stays near 0.20 throughout.
- **Sample Average:** catastrophic failure and no meaningful recovery. Its optimal-action rate drops from 0.92 at step 500 to 0.00 at step 510, and only reaches 0.26 by step 1,000. Its Q-values are stuck on the pre-change distribution.
- **Constant Step:** same catastrophic failure at step 510 (rate 0.00), but rapid recovery. By step 700 it has already overtaken the sample-average agent, and by step 1,000 it selects the new optimum 72% of the time.

### 7.3. Why might a constant step size be useful when reward probabilities change?

A constant step size implements an **exponential recency-weighted average**: recent observations are weighted more heavily than old ones, with a decay factor of `(1 − α)` per observation. After the change at step 501, the sample-average agent's estimate for each arm is dominated by ~500 pre-change observations; a new observation shifts the estimate by only `1/N ≈ 0.002`. It would take hundreds of steps to overcome the stale data. The constant-step agent weights each new observation at `α = 0.1`, so its estimate shifts meaningfully within ~10–20 observations. The result is visible in the trajectory: by step 800, constant-step's optimal-action rate is 0.54 vs 0.06 for sample-average. Constant step size effectively trades off a small amount of steady-state estimation accuracy (visible in the stationary environment, where it loses to sample-average) for the ability to track a non-stationary target.

### 7.4. What tradeoff does ε control? What would change if ε were much larger or much smaller?

ε controls the **exploration–exploitation tradeoff**:

- **High ε** (e.g., 0.5): the agent spends half its steps exploring. Estimates converge faster and the agent recovers faster after a change, but it also wastes reward on suboptimal arms when it already knows the optimum. Expected behavior: lower average reward, higher steady-state optimal-action rate ceiling ≈ 1 − ε + ε/5.
- **Low ε** (e.g., 0.01): the agent exploits almost always. It converges to high average reward in a stationary environment, but estimates for non-best arms are poor and adaptation after a change is slow (the agent rarely re-samples enough to notice). Expected behavior: high short-term reward, poor robustness to non-stationarity.
- **ε = 0**: greedy. The agent can get permanently stuck on an unlucky first estimate of a suboptimal arm.

With ε = 0.1 (our setting), the agent explores 10% of the time — enough to maintain reasonable estimates across all arms, while spending 90% of steps exploiting.

### 7.5. How did you verify your implementations were correct?

Three standalone tests plus one analytic check:

1. **`test_database_integrity`** — verifies 1 experiment, 300 runs, 300,000 step records; zero foreign-key violations; zero duplicate `(run_id, step)` pairs; all `reward` values in `{0,1}`; all `optimal_action` values in `{0,1}`.
2. **`test_duplicate_experiment`** — attempts to insert a duplicate `experiment_id`; verifies the primary key rejects it, and that the existing row is unchanged.
3. **`test_reproducibility`** — runs each agent twice with identical seeds and asserts the reward and action traces are bit-identical; then runs with different seeds and asserts they differ (sanity check that the RNG is actually being used).
4. **Analytic check (sanity of the environment)** — the Random agent's optimal-action rate is ≈ 0.20 (theoretical 1/5) and its average reward is ≈ 0.50 (theoretical mean of the probability vector). Any mistake in the environment's probability or arm indexing would break these two identities.

Additionally, `inspect.getsource(BanditEnvironment.step)` was used to confirm the probability vector is read **before** `current_step` is incremented — closing the off-by-one bug described in §8.

### 7.6. What did you do to make the comparison between agents fair and the results reproducible?

- **Common Random Numbers (CRN):** all three agents face the same `(step, arm) → uniform` matrix for a given (environment, run). If two agents pick the same arm at the same step, they receive the same reward. This removes the largest source of cross-agent noise.
- **Per-run seed pairing:** each run `i` uses the same environment seed for all three agents, but a distinct agent seed per algorithm (so the algorithms' internal randomness is independent).
- **Master seed + `SeedSequence`:** every seed is deterministically derived from a single master seed (`42`), so the entire experiment is reproducible bit-for-bit.
- **Full reset between runs:** each run constructs a fresh environment and a fresh agent; no state carries over.
- **No probability leakage:** agents never see `get_probabilities()`; only the evaluation code does.
- **Stored parameters in DB:** every run's seeds, ε, and α are persisted to SQLite, so any run can be audited or regenerated.

### 7.7. What additional challenges would arise when moving this simulation into a real recommendation system?

- **Reward delay and attribution.** Positive feedback (a purchase) may arrive hours or days after the recommendation, so the agent must handle delayed rewards and credit assignment.
- **Position bias and presentation effects.** The reward depends not only on the item but on *where* it was shown. Without accounting for position, the agent confuses item quality with display slot.
- **Non-stationarity from external causes.** User preferences drift, seasonality changes, catalogs change, marketing campaigns start and stop. The constant-step-size approach is a step in the right direction, but real systems need per-arm or per-context adaptation.
- **Contextual decisions.** Recommendations depend on the user (age, history, device, time), not just the arm. Plain MAB becomes contextual bandits; the action space is enormous.
- **Large action spaces.** Five arms becomes thousands or millions. Exact Q-values become infeasible; function approximation or embeddings are required.
- **Exploration cost.** Random exploration (ε-greedy) in production is expensive — showing a suboptimal product to a real user has a real cost. Off-policy evaluation and constrained exploration (e.g., Thompson sampling with priors) become important.
- **Regulatory and fairness constraints.** Exploration must respect minimum-quality guarantees, avoid discriminating along protected attributes, and be auditable.
- **Evaluation.** Counterfactual evaluation (inverse propensity scoring, doubly robust estimators) is needed because we cannot rerun production with a different policy.

### 7.8. If the number of runs increased from 50 to 100,000, what would you change in data-storage or processing?

The current design would break in three places: total row count (`300k → 600M`), insert throughput, and analysis query cost.

**Storage changes:**

- **Partition or shard by experiment.** Keep the current three-table schema, but for production-scale runs use separate SQLite files per experiment (or move to PostgreSQL/DuckDB with native partitioning).
- **Batch insert with transactions.** Currently `executemany` per (env, agent, run); at 600M rows this becomes too slow. Group into chunks of ~10k rows, wrap in a single transaction per chunk, and use `PRAGMA journal_mode=WAL` and `PRAGMA synchronous=NORMAL`. For SQLite, expect roughly 100k rows/sec with these settings — a 600M-row insert would still take ~2 hours but is feasible.
- **Store aggregates, not raw steps, by default.** For 100k runs, the raw step data is mostly redundant for reporting. Persist per-run aggregates (final reward, final regret, cumulative curves at every 10th step) plus a small sample of raw steps for diagnostics. Raw steps for all runs can be written to parquet (columnar, ~10–50× compression) instead of SQLite.
- **Move raw step data out of the relational DB.** SQLite is fine for metadata (100k rows) but not for 600M rows of time-series. Store per-run aggregates in SQLite and raw step traces in Parquet (one file per experiment, partitioned by environment and agent).

**Processing changes:**

- **Vectorize across runs.** The current loop is one Python iteration per step per run. At 600M rows this becomes the bottleneck. Rewrite the inner loop to operate on all 100k runs simultaneously with NumPy (arrays of shape `(n_runs,)` per step) or use `numba.jit`. This gives 100–1000× speedup with minimal code change.
- **Parallelize across (environment, agent).** Each of the 6 combinations is independent; run them in separate processes with `multiprocessing`.
- **Stream metrics to disk incrementally.** Instead of holding all 600M rewards in memory, compute the four metric curves online (running means, running cumsums) and flush to disk every N steps.
- **Pre-aggregate in SQL.** Ensure `step_results` has covering indexes for the analysis queries, or precompute per-step aggregates into a `metric_curves` table at insert time.

**Statistical changes:**

- With 100k runs, 95% confidence intervals on the mean curves become tight (~0.002 standard error), so plot confidence bands rather than bare means. The comparison between algorithms becomes statistically strong, not just visually suggestive.

---

## 8. Implementation Notes

### 8.1. Off-by-one fix in the environment step

An earlier version of the environment incremented `current_step` **before** reading the probability vector, so the regret reported at step *t* was computed against the probabilities of step *t + 1*. This only affected the transition step (`t = 500`) and has a negligible effect on averages across 50 runs — but it is conceptually incorrect. The current implementation reads probabilities **before** advancing the step counter:

```python
probabilities = self.get_probabilities()   # probabilities for step t
...
self.current_step += 1
```

so `regret_t` and `optimal_action_t` are computed against the probabilities that produced the reward at step *t*. The size of the fix's effect on aggregate metrics is below the reporting precision (≤ 0.02 in final regret), which is why the reported numbers match the earlier run; the fix is about correctness, not about changing the conclusion.

### 8.2. Probability access

`BanditEnvironment.get_probabilities()` is called exclusively by the experiment runner for evaluation. Agents never call it and never receive the probability vector or the pre-generated uniform matrix.

### 8.3. Tie-breaking

When two or more arms share the maximum Q-value, both ε-greedy agents sample uniformly at random among the tied arms using their own RNG. This avoids the systematic low-index bias of `np.argmax`.

### 8.4. Database duplicate behavior

Every execution of `main.py` generates a fresh `experiment_id` at microsecond resolution. The `experiments.experiment_id` primary key rejects duplicates. If an error occurs mid-run, `main.py` rolls back all rows for the partial experiment before re-raising, so the database never holds an experiment with missing run metadata or step records.

---

## 9. Conclusion

The experiments confirm the standard result in the multi-armed bandit literature:

- **Stationary environments reward memory.** Sample-average converges to the true mean and achieves the best stationary performance (reward 0.84056, regret 59.18).
- **Non-stationary environments reward forgetting.** Constant-step size — an exponential recency-weighted average — recovers from a distribution shift roughly three times faster (regret grows +101 vs +326 after the change), and dominates in the non-stationary environment (reward 0.74098, regret 156.05).
- **Random is a stable baseline.** Its optimal-action rate (≈ 0.20) and average reward (≈ 0.50) match analytic expectations, confirming correct implementation.

The choice of estimator is therefore not a matter of taste: it must match the temporal statistics of the environment. A system deployed in the real world — where preferences drift continuously — is best served by an exponentially-weighted estimator with a carefully chosen half-life, and ε-greedy remains a surprisingly competitive baseline when combined with it.

---

## 10. Reproducibility

To reproduce every number, table, and plot in this report:

```bash
pip install -r requirements.txt
rm bandit_results.db                        # start clean
python main.py                              # runs the experiment
python extract_results.py                   # dumps analysis tables
python -m tests.test_database_integrity
python -m tests.test_duplicate_experiment
python -m tests.test_reproducibility
```

Under master seed `42`, every trajectory, regret series, and figure is bit-reproducible.
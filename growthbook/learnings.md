# GrowthBook Learning Notes

## Goals

- **A) Practice writing Python** — syntax, idioms, standard library, pandas
- **B) Bolster Bayesian statistics understanding** — priors, posteriors, credible intervals
- **C) Build real data pipelines** — Kafka, PostgreSQL, streaming vs. batch

---

## Part 1: The GrowthBook Stats Engine (`packages/stats/`)

### What it is

A Python package (`gbstats`) that performs A/B test statistical analysis. Published to PyPI at version `0.8.0`. Stack: Python 3.9+, pandas, scipy, numpy, pydantic. Managed via Poetry.

### How it runs

A long-running Python subprocess pool (1–4 processes). The Node.js back-end communicates with it via newline-delimited JSON over stdin/stdout. Each request is `{id, data}`, each response is `{id, results, time}`.

An alternative HTTP mode exists: if `EXTERNAL_PYTHON_SERVER_URL` is set, the back-end POSTs to that endpoint instead of using the subprocess pool.

**Relevant env vars:**
- `GB_STATS_ENGINE_POOL_SIZE` — max concurrent Python processes (default: 4)
- `GB_STATS_ENGINE_MIN_POOL_SIZE` — min warm processes (default: 1)
- `GB_STATS_ENGINE_TIMEOUT_MS` — request timeout (default: 300,000ms)
- `EXTERNAL_PYTHON_SERVER_URL` — external Python API endpoint (optional)

### Core data flow

1. Back-end sends raw SQL query results + metric/analysis config as JSON
2. `process_experiment_results()` in `gbstats/gbstats.py` orchestrates everything
3. Groups data by dimension → creates typed `Statistic` objects → runs the appropriate test class → packages results
4. Returns `ExperimentMetricAnalysis` per metric + optional `BanditResult` for weight reallocation

### Directory structure

```
packages/stats/
├── gbstats/
│   ├── gbstats.py                # Core orchestration and data processing
│   ├── utils.py                  # SRM check, statistical helpers
│   ├── messages.py               # Error message constants
│   ├── gen_notebook.py           # Jupyter notebook generation
│   ├── models/
│   │   ├── settings.py           # Input configuration models (Pydantic)
│   │   ├── results.py            # Output result models (Pydantic)
│   │   ├── statistics.py         # Statistical computation objects (frozen dataclasses)
│   │   └── tests.py              # Test base classes + effect moment calculations
│   ├── bayesian/
│   │   ├── tests.py              # EffectBayesianABTest class
│   │   └── bandits.py            # Thompson sampling bandit algorithms
│   ├── frequentist/
│   │   └── tests.py              # T-tests (two-sided, one-sided, sequential)
│   ├── power/
│   │   └── midexperimentpower.py # Mid-experiment power calculations
│   └── devtools/
│       └── simulation.py         # Simulation utilities for testing
├── tests/                        # pytest suite
├── pyproject.toml                # Poetry dependencies
├── requirements.txt
└── package.json                  # pnpm scripts (test, lint, setup, build)
```

### Key files

| File | Purpose |
|------|---------|
| `gbstats/gbstats.py` | Main orchestration — `process_experiment_results()` entry point |
| `gbstats/models/settings.py` | Input Pydantic models: `AnalysisSettingsForStatsEngine`, `MetricSettingsForStatsEngine`, `BanditSettingsForStatsEngine` |
| `gbstats/models/results.py` | Output Pydantic models: `ExperimentMetricAnalysis`, `BayesianVariationResponse`, `FrequentistVariationResponse` |
| `gbstats/models/statistics.py` | Frozen dataclasses: `SampleMeanStatistic`, `ProportionStatistic`, `RatioStatistic`, `RegressionAdjustedStatistic` |
| `gbstats/models/tests.py` | `BaseABTest` — aggregates paired stats, computes effect moments |
| `gbstats/frequentist/tests.py` | `TwoSidedTTest`, `SequentialTwoSidedTTest`, one-sided variants |
| `gbstats/bayesian/tests.py` | `EffectBayesianABTest` |
| `gbstats/bayesian/bandits.py` | `BanditsSimple`, `BanditsRatio`, `BanditsCuped` |
| `packages/back-end/scripts/stats_server.py` | Subprocess entry point — reads JSON from stdin, writes to stdout |

---

## Part 2: Statistical Methods

### Bayesian A/B test details (`bayesian_stats.py`)

Built from scratch with scipy. Reproduces GrowthBook's +10.8% lift from first principles.

- **Prior**: Flat (improper) — posterior = likelihood
- **Likelihood**: Normal, from sample statistics
- **Posterior**: Closed-form normal posterior (conjugate Normal/Normal)
- **Delta method**: Variance of relative lift via `Var(M/D)` approximation
- **Credible interval**: Gaussian quantiles at alpha/2
- **Chance to win**: `norm.sf(0, loc=lift, scale=SE)` — P(lift > 0)
- **Risk metrics**: Expected loss via truncated normal means
- **Matplotlib**: Posterior distribution plot with 95% CI shaded

### Frequentist A/B test details (`frequentist_stats.py`)

Built from scratch with scipy. Same data pipeline as Bayesian, different interpretation step.

- **Welch's t-test**: Does NOT assume equal variance between groups
- **t-statistic**: `lift / SE` — normalizes to a standard t-distribution
- **Welch-Satterthwaite df**: Accounts for potentially different group variances
- **p-value**: `2 * (1 - t.cdf(abs(t_stat), df))` — two-sided, matches GrowthBook exactly
- **95% CI**: `lift ± t.ppf(0.975, df) * SE`
- **Matplotlib**: Null t-distribution with observed t-stat and p-value tails shaded

### Variance reduction
- **CUPED** — uses pre-experiment covariate to reduce variance (`RegressionAdjustedStatistic`)
- **Post-stratification** — stratifies by pre-computed dimensions

### Diagnostics
- **SRM (Sample Ratio Mismatch)** — chi-squared test against expected traffic split
- **Mid-experiment power analysis** — estimates additional sample needed for target power

### Statistic types

All are frozen dataclasses inheriting from a base `Statistic` class with `mean`, `variance`, `stddev` properties.

| Class | Use case |
|-------|---------|
| `SampleMeanStatistic` | Continuous metrics (mean, variance, sum) |
| `ProportionStatistic` | Binomial metrics (mean = count/n) |
| `RatioStatistic` | Ratio metrics (numerator/denominator with covariance) |
| `RegressionAdjustedStatistic` | CUPED — pre/post period stats with covariance |
| `RegressionAdjustedRatioStatistic` | CUPED for ratio metrics |
| `ScaledImpactStatistic` | Revenue/scaled impact calculations |

---

## Part 3: Kafka Integration

### Goal

Build an end-to-end pipeline so that exposures generated by `experiment.py` flow through Kafka into PostgreSQL, where GrowthBook can query them and run Bayesian analysis in the UI.

### Why GrowthBook can't read Kafka directly

GrowthBook queries a SQL data source — Kafka is a streaming transport, not a queryable store. PostgreSQL is the handoff point between the stream and GrowthBook.

### Full architecture

```
experiment.py (Python SDK)
  → on_experiment_viewed callback
    → KafkaProducer → "experiment-events" topic
      → KafkaConsumer (consumer.py)
        → writes to PostgreSQL
          → GrowthBook connects to PostgreSQL as data source
            → SQL query → gbstats (Bayesian analysis)
              → results displayed in UI
```

### PostgreSQL schema

- `exposures` — `(user_id, experiment_id, variation_id, timestamp)` — who saw what
- `metrics` — `(user_id, value, timestamp)` — outcome data for Bayesian lift calculation

### Build order

- [x] **Step 1 — Infrastructure:** Add Kafka + Zookeeper + PostgreSQL to Docker (`docker-compose.yml`)
- [x] **Step 2 — Producer:** Update `experiment.py` callback to publish to Kafka instead of `print()`
- [x] **Step 3 — Consumer:** Build `consumer.py` — reads from Kafka topic, writes rows to PostgreSQL
- [x] **Step 4 — GrowthBook config:** Connect GrowthBook UI to PostgreSQL data source, configure SQL queries for exposures + metrics, run analysis

### Generated synthetic metrics in pandas

Built `generate_metrics.py` using pandas:
- Start with a DataFrame of `user_id` values (same users from `experiment.py`)
- Simulate an outcome value per user (e.g. purchase amount, binary conversion)
- Treatment effect baked in — variation "1" users get a slightly higher mean
- Write the result to the `metrics` table in PostgreSQL (using `df.to_sql()`)

Good pandas practice: DataFrame construction, random sampling, conditional column logic, `to_sql()`.

---

## Part 4: Learnings

Concepts picked up along the way, organized by topic.

### Python packages & tooling

**Two packages, two install methods.** GrowthBook has two separate Python packages:

| Package | Purpose | Install for hacking |
|---------|---------|-------------------|
| `growthbook` (SDK) | Evaluates flags, buckets users in app code | `pip install growthbook` (PyPI-only, not in monorepo) |
| `gbstats` (stats engine) | Runs A/B test math (t-tests, Bayesian, bandits) | `poetry install` inside `packages/stats/` |

**The key rule:** Use `poetry install` so your local edits take effect immediately (editable mode). Pulling from PyPI gives a frozen snapshot that ignores your changes. `pip install growthbook` is only needed for `experiment.py` (the SDK client), which is genuinely not in the monorepo.

### Feature flags & experimentation

**Variation key vs. variation value — two audiences, same object.** In traditional data science A/B testing you work **downstream** of assignment. By the time data lands in your warehouse it's just `(user_id, experiment_id, variation, timestamp)` — what that variation *delivered* was already handled by engineering.

In a feature flag SDK, the split is explicit because **the SDK is the product code**:
- `result.key` → `"1"` — which bucket (what you log to analytics)
- `result.value` → `{"cta": "Buy Now", "color": "blue"}` — what the app actually renders

Engineering and data science are touching the same object from different angles. `gbstats` only needs variation assignment + metric outcomes — the payload is an application concern, not a statistics concern.

### Security

**SQL injection via f-strings.** Never interpolate Python values directly into SQL strings (f-strings, `.format()`, `%` string formatting). If user-controlled data reaches the query, an attacker can break out of the string and run arbitrary SQL.

psycopg2's safe pattern — values as a separate tuple, `%s` as placeholders:
```python
cur.execute("INSERT INTO t (col1, col2) VALUES (%s, %s)", (val1, val2))
```
psycopg2 escapes the values before they touch the SQL string. The query structure and the data never mix.

### Pandas  gotchas

**`.sum()` vs `.agg()` — different APIs for column selection.** `.agg()` accepts column names as arguments (e.g., `.agg(total=('metric_0', 'sum'))`), but `.sum('metric_0')` does NOT select a column — the string argument is interpreted as the `axis` parameter, causing a boolean coercion error. Always select the column first with bracket notation: `df.groupby(...)['metric_0'].sum()`.

**`.loc` vs `.iloc` — label vs position.** `.loc['0']` selects by index *label* (the actual value). `.iloc[0]` selects by integer *position* (first row). After a `.groupby().agg()`, the group column becomes the index, so use `.loc['variation_id']` to grab a specific group's row. `iloc` stands for "integer location", not "index location" — `.loc` is the one that uses index labels.

### Matplotlib

**Use the OOP approach with `plt.subplots()`.** Matplotlib has two interfaces: stateful (`plt.plot()`) and OOP (`ax.plot()`). Always use OOP — it's explicit and scales to multiple subplots.

```python
fig, ax = plt.subplots(figsize=(10, 5))     # factory — creates figure + axes
fig, (ax1, ax2) = plt.subplots(1, 2)        # two charts side by side
```

**Three objects, three roles:**
- `fig` — canvas-level: `fig.suptitle()`, `fig.tight_layout()`, `fig.savefig()`
- `ax` — chart-level: `ax.plot()`, `ax.fill_between()`, `ax.set_title()`, `ax.set_xlabel()`
- `plt.show()` — display (blocking call, keeps window open until closed)

**Common gotcha:** `fig.show()` is non-blocking — window opens and immediately closes when script ends. Always use `plt.show()` instead.

**Setter pattern:** Use `ax.set_title()`, `ax.set_xlabel()`, `ax.set_xlim()` — the `set_` prefix is for setting, bare name is for getting.

### Data pipeline patterns

**Batch vs. stream — two pipeline patterns.**

| Pattern | Example | Behavior | Latency |
|---------|---------|----------|---------|
| Streaming | `consumer.py` | Poll one message → insert one row → repeat forever | Milliseconds–seconds |
| Batch | `generate_metrics.py` | Load everything → transform in memory → write all at once → exit | Minutes–hours |

Most real-world systems use both. Streaming handles real-time event capture (exposures), batch handles derived/enriched datasets (metrics). This project uses both patterns end-to-end.

**Apache Flink — stream processing engine.** If Kafka is the pipe that moves data, Flink is the brain that processes data as it flows through. Flink sits on top of Kafka and computes aggregations continuously, without waiting for a batch job.

Data science analogy: Kafka is like a real-time DataFrame that keeps appending rows. Flink is like running `.groupby().agg()` on that stream continuously, emitting results as new data arrives.

| Approach | Tool | Latency |
|----------|------|---------|
| Batch | pandas + cron / `generate_metrics.py` | Minutes to hours |
| Micro-batch | Spark Streaming | Seconds to minutes |
| True streaming | Flink | Milliseconds to seconds |

Flink doesn't use pandas — it has its own API (Java/Scala natively, Python via **PyFlink**). PyFlink's Table API looks similar to pandas (selects, filters, group-bys) but operates on unbounded streams, not in-memory DataFrames.

In the context of this project: with Flink, `generate_metrics.py` could be replaced by a streaming computation that aggregates metrics as events arrive, rather than in a separate batch step. Flink replaces the batch compute, not Kafka — Kafka still moves the data.

Relevant use cases in experimentation platforms:
- Real-time aggregation of experiment events (counts, sums, averages per variant)
- Sessionization — stitching click streams into user sessions on the fly
- Anomaly detection — flagging metrics that diverge mid-experiment (SRM checks)

---

## Part 5: Up Next

### Deep dive into GrowthBook's 3 statistic types

Understand how proportions, means, and ratios each flow through the stats engine differently. Focus on `ProportionStatistic`, `SampleMeanStatistic`, and `RatioStatistic` in `models/statistics.py` — how variance is computed, how the delta method adapts for ratios, and when each type applies.

### Contribute to GrowthBook — fix an open issue

Take another pass at `open_issues.md`, find a good-first-issue, and submit a PR. Goal: demonstrate ability to navigate a production codebase and contribute upstream.

### Bonus: MCMC Bayesian analysis (lower priority)

Reimplement the Bayesian A/B test using MCMC sampling (PyMC or Stan) instead of the closed-form conjugate update. Same data, same question, but let the sampler find the posterior numerically. Goal: understand when MCMC is necessary vs. overkill, and see firsthand that it converges to the same answer as the closed-form solution for this simple case.

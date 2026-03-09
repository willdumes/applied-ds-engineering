# MLflow Learning Notes

## Goals

- **A) Bridge the DS-to-MLE gap** -- understand model lifecycle tooling from build to deploy
- **B) Practice Python** -- scikit-learn, pandas, matplotlib, MLflow APIs
- **C) Build a real ML project end-to-end** -- train, track, register, and serve a model
- **D) Professional development** -- deepen hands-on ML engineering skills alongside DS management experience

---

## Plan

### Phase 1: Explore & set up

- [x] Fork MLflow repo, create hacking scratchpad
- [x] Explore repo structure and Docker Compose stack
- [x] Stand up the Docker stack (Postgres + RustFS + MLflow server on port 5433/9000/5000)
- [x] Run the wine quality example (3 runs with different hyperparams, visible in UI)
- [x] Parse Strava FIT data with fitparse + pandas (10,336 rows from Napa marathon)
- [x] Walk through train.py line by line -- understand the full tracking flow
- [ ] Walk through the MLflow quickstart guides

### Phase 2: Build a model with Strava data

- [x] Export running data from Strava (bulk export -- .fit.gz files)
- [ ] EDA in pandas + matplotlib
- [ ] Train a model (e.g., predict race finish time from training features)
- [ ] Track experiments in MLflow (params, metrics, artifacts)
- [ ] Register best model in Model Registry
- [ ] Serve model locally via `mlflow models serve`

### Phase 3: GenAI side of MLflow (with local Qwen3.5 30B)

- [ ] Set up MLflow tracing with local Qwen3.5 via Ollama
- [ ] Log and inspect LLM traces in the MLflow UI
- [ ] Use MLflow Evaluate to score LLM outputs (relevance, toxicity, etc.)
- [ ] Explore Prompts and AI Gateway features

### Phase 4: Go deeper

- [ ] Find and fix an open issue in the MLflow repo
- [ ] Optional: deploy model to a simple endpoint (Docker or local Flask)

---

## Part 1: MLflow repo structure

### What it is

An open-source platform for the complete ML lifecycle. Version 3.10.1.dev0. Stack: Python 3.10+, Flask, SQLAlchemy, React UI. Owned by Databricks, Apache 2.0 licensed.

### Key directories

```
mlflow/
├── mlflow/              # Core Python package
│   ├── tracking/        # Experiment tracking (logging params, metrics, artifacts)
│   ├── models/          # Model packaging (MLmodel format, flavors)
│   ├── store/           # Backend + artifact storage implementations
│   ├── server/          # Flask REST API + React UI (server/js/)
│   ├── sklearn/         # scikit-learn integration (autolog, model logging)
│   ├── pyfunc/          # Generic Python model interface
│   ├── evaluation/      # Model evaluation framework
│   ├── projects/        # Reproducible ML code packaging
│   └── [integrations]/  # pytorch, tensorflow, xgboost, langchain, etc.
├── examples/            # Example scripts (sklearn, keras, etc.)
│   └── sklearn_elasticnet_wine/  # Classic example -- ElasticNet on wine quality
├── docker-compose/      # Docker stack: Postgres + RustFS (S3) + MLflow server
├── docker/              # Dockerfiles for MLflow images
├── tests/               # Test suite
└── docs/                # Documentation source
```

### Docker Compose stack

Three services, same pattern as the GrowthBook setup:

| Service | Image | Role | Port |
|---------|-------|------|------|
| `postgres` | `postgres:15` | Metadata store (experiments, runs, params, metrics) | 5432 |
| `storage` (RustFS) | `rustfs/rustfs` | S3-compatible artifact store (model files, plots) | 9000 |
| `mlflow` | `ghcr.io/mlflow/mlflow` | Tracking server + UI | 5000 |

Plus a one-shot `create-bucket` container (AWS CLI) that creates the S3 bucket on first run.

### Parallel to GrowthBook

| Concept | GrowthBook | MLflow |
|---------|-----------|--------|
| Core question | "Did the change work?" | "Which model is best?" |
| Metadata store | MongoDB | PostgreSQL (or SQLite) |
| Artifact store | N/A | S3 / local filesystem |
| Key abstraction | Experiment (A/B test) | Experiment (set of runs) |
| Versioning | Feature flags | Model Registry |
| Stats engine | gbstats (Python) | User brings their own (sklearn, etc.) |

---

## Part 2: Key MLflow concepts

### Tracking

The core workflow: log everything about a model training run so you can compare and reproduce.

```python
import mlflow

with mlflow.start_run():
    mlflow.log_param("alpha", 0.5)       # hyperparameter
    mlflow.log_metric("rmse", 0.82)      # evaluation metric
    mlflow.sklearn.log_model(model, name="model")  # the model itself
```

- **Experiment** -- a named group of runs (like a folder)
- **Run** -- one execution of training code (has params, metrics, artifacts, tags)
- **Artifact** -- any file: model pickle, plot PNG, CSV, etc.

### Autolog

One line replaces all the manual `log_param`/`log_metric` calls:

```python
mlflow.sklearn.autolog()
model.fit(X_train, y_train)  # MLflow intercepts this and logs everything
```

Logs: hyperparameters, training metrics, model signature, even feature importance plots.

### Model Registry

Version and stage-manage models:
- Log a model during a run
- Register it with a name (e.g., "StravaRacePredictor")
- Promote versions through stages (None -> Staging -> Production)
- Query by stage for deployment

### Model serving

```bash
mlflow models serve -m "models:/StravaRacePredictor/Production" --port 5001
```

Spins up a REST API that accepts JSON input and returns predictions.

---

## Part 3: Strava running model brainstorm

### Data available from Strava

Strava bulk export gives a CSV of all activities with:
- Distance, duration, elevation gain
- Average/max heart rate, average pace
- Activity type (run, race, workout)
- Date, time, weather (maybe via external API)
- Training load / relative effort (if available)

### Model ideas

1. **Race time predictor** -- predict 10K/half marathon finish time from recent training
   - Features: weekly mileage, long run distance, avg pace, elevation, rest days
   - Target: actual race finish time
   - Why: tangible, explainable, built on real personal data

2. **Injury risk predictor** -- predict weeks where performance drops
   - Features: training load ramp rate, rest days, intensity distribution
   - Target: binary (performance drop in next 2 weeks)
   - Why: interesting classification problem, practical utility

3. **Optimal training plan** -- cluster training weeks by pattern, find what works
   - Unsupervised: k-means on weekly training features
   - Then correlate clusters with race performance
   - Why: shows breadth (clustering + analysis), less prediction-focused

### Recommendation: Start with #1 (race time predictor)

It's the most concrete, has a clear target variable, uses regression (familiar), and makes a great story. Can always add #2 or #3 later.

### Models to try

- **ElasticNet** — linear baseline, fast, interpretable, good for comparison
- **XGBoost** — gradient-boosted trees, typically best for tabular data, handles non-linear relationships and feature interactions without manual engineering

---

## Part 4: Learnings

### Docker & infrastructure

**Port mapping has two planes.** `"5433:5432"` means host port 5433 maps to container port 5432. Services inside the Docker network still talk on 5432 -- the host port only matters when *you* connect from your Mac. GrowthBook's Postgres already had 5432, so MLflow's Postgres goes on 5433.

**One-shot init containers.** The `create-bucket` container runs once (creates the S3 bucket in RustFS), then exits with code 0. `restart: "no"` ensures Docker doesn't restart it. Same pattern as running a database migration on startup.

**`.env` vs `.venv`.** `.env` is a plain text file of environment variables for Docker Compose. `.venv` is a Python virtual environment. Completely unrelated despite the similar names.

### FIT file format

**Binary telemetry from GPS watches.** FIT (Flexible and Interoperable Data Transfer) is Garmin's binary format. One record per second with GPS, heart rate, speed, altitude, power, cadence. ~26 bytes per data point (vs ~200+ bytes as CSV text).

**Message types in a FIT file.** `record` (per-second telemetry), `lap` (split summaries), `session` (whole-activity summary), `event` (start/stop). Each has different fields -- can't mix into one DataFrame.

**GPS stored as semicircles.** Convert to degrees: `degrees = semicircles * (180 / 2^31)`.

**Parse to pandas:** `fitparse.FitFile` reads the binary, then dict comprehension + `pd.DataFrame()` gives you a table.

**Scale limits of list comprehensions + pandas.** List comprehensions load everything into RAM — at 200GB you'd OOM before `pd.DataFrame()` finishes. `fitparse` is a streaming parser (yields one message at a time), so you can write to Parquet in batches to avoid holding everything in memory. Rule of thumb: pandas works up to ~5-10GB (fits in RAM), chunking (`pd.read_csv(chunksize=N)`) buys more headroom, PySpark/Dask for 100GB+.

### Regularization (ElasticNet)

**Regularization prevents overfitting by penalizing large coefficients during training.** The prediction formula is plain linear regression (`ŷ = β₀ + β₁x₁ + ...`), but the loss function adds a penalty: `α * l1_ratio * Σ|βⱼ|` (L1/Lasso — can zero out features) + `α * (1-l1_ratio) * Σβⱼ²` (L2/Ridge — shrinks everything). L1 is automatic feature selection built into training. `alpha` = penalty strength, `l1_ratio` = L1/L2 mix.

### Python error handling

**try/except without sys.exit() doesn't stop execution.** If the except block only logs the error, the script continues and crashes later with a less helpful error (e.g., `NameError: name 'data' is not defined`). Always add `sys.exit(1)` to fail fast with a clear message. The wine example's `train.py` has this bug — `sys` is imported but unused for this purpose.

### Flask & web architecture

**Flask = Python web framework.** Turns Python functions into HTTP endpoints. MLflow's tracking server is a Flask app serving both the REST API and the React UI (as static files). "Static" means the JS files don't change at request time -- all interactivity runs in the browser.

**Postgres for metadata, S3 for blobs.** Same architectural split used by GitHub, Spotify, etc. Structured data (params, metrics) in SQL, large files (models, plots) in object storage.

---

## Part 5: Up Next

- Walk through train.py line by line (ElasticNet, train/test split, MLflow logging)
- Start Strava EDA: parse all activities, clean NaNs, engineer features
- Build first regression model: predict race time from training block features
- Track experiments in MLflow and compare runs in the UI

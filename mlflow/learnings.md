# MLflow Learning Notes

## Goals

- **A) Practice Python** -- scikit-learn, pandas, matplotlib, MLflow APIs
- **B) Build a real ML project end-to-end** -- train, track, register, and serve a model


---

## Plan

### Phase 1: Set up Docker and explore repo

- [x] Fork MLflow repo, create dev scratchpad
- [x] Explore repo structure and Docker Compose stack
- [x] Stand up the Docker stack (Postgres + RustFS + MLflow server on port 5433/9000/5000)

### Phase 2: ML models (train.py intro + Strava deep dive)

- [x] Run the wine quality example (3 runs with different hyperparams, visible in UI)
- [x] Walk through train.py and deep dive into ElasticNet Lasso/Ridge regularization
- [x] Parse Strava FIT data with fitparse + pandas (10,336 rows from Napa marathon)
- [x] EDA in pandas
- [x] Train a linear model (e.g., predict pace from training features)
- [x] Track experiments in MLflow (params, metrics, artifacts)
- [x] Train an advanced model to get high predictability (e.g. XGBoost)
- [ ] Register best model in Model Registry
- [ ] Serve model locally via `mlflow models serve`

### Phase 3: GenAI (Data Science AI Agent with Qwen 3.5)

- [x] Set up MLflow tracing with local Qwen3.5 via Ollama
- [x] Log and inspect LLM traces in the MLflow UI
- [ ] Use MLflow Evaluate to score LLM outputs (relevance, toxicity, etc.)
- [ ] (bonus) Build a Data Science AI agent
- [ ] (bonus) Build a RAG system: embed Strava run summaries with `embeddings.create()`, retrieve relevant context per question, pass to `chat.completions.create()`

### Phase 4: Go deeper

- [ ] Find and fix an open issue in the MLflow repo
- [ ] Optional: deploy model to a simple endpoint (Docker or local Flask)

---

## Part 1: MLflow set up with Docker and repo overview

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

## Part 2: ML models

### Key MLflow concepts

#### Tracking

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

#### Autolog

One line replaces all the manual `log_param`/`log_metric` calls:

```python
mlflow.sklearn.autolog()
model.fit(X_train, y_train)  # MLflow intercepts this and logs everything
```

Logs: hyperparameters, training metrics, model signature, even feature importance plots.

#### Model Registry

Version and stage-manage models:
- Log a model during a run
- Register it with a name (e.g., "StravaRacePredictor")
- Promote versions through stages (None -> Staging -> Production)
- Query by stage for deployment

#### Model serving

```bash
mlflow models serve -m "models:/StravaRacePredictor/Production" --port 5001
```

Spins up a REST API that accepts JSON input and returns predictions.

### Strava running model brainstorm

#### Data available from Strava

Strava bulk export gives a CSV of all activities with:
- Distance, duration, elevation gain
- Average/max heart rate, average pace
- Activity type (run, race, workout)
- Date, time, weather (maybe via external API)
- Training load / relative effort (if available)

#### Model ideas

1. **Realtime pace predictor**
   - Features: shoes, elevation, humidity
   - Target: speed
   - Why: tangible, explainable, built on real personal data

2. **Injury risk predictor**
   - Features: training load ramp rate, rest days, intensity distribution
   - Target: binary (performance drop in next 2 weeks)
   - Why: interesting classification problem, practical utility

3. **Optimal training plan**
   - Unsupervised: k-means on weekly training features
   - Then correlate clusters with race performance
   - Why: shows breadth (clustering + analysis), less prediction-focused

#### Models to try

- **ElasticNet** -- linear baseline, fast, interpretable, good for comparison. R2 ~ 0.44, RMSE ~ 0.48 on 75 runs
- **GBRT** -- sklearn gradient-boosted trees, no regularization on leaf weights. R2 ~ 0.67, RMSE ~ 0.36 on 75 runs
- **XGBoost** -- gradient-boosted trees with L1/L2 regularization and early stopping. R2 ~ 0.71, RMSE ~ 0.34 on 75 runs (best)

---

## Part 3: GenAI — LLM Tracing & Evaluation with Qwen 3.5

### LLM chat API fundamentals

Every major LLM provider (OpenAI, Anthropic, Ollama) uses the same core abstraction: you send a list of **messages**, each tagged with a **role**, and the model returns a completion.

#### The four roles

| Role | Purpose | Example |
|------|---------|---------|
| `system` | Standing instructions sent with every API call. Sets behavior, persona, constraints. Not persisted by the model. | "You are a running coach. Keep responses under 200 words." |
| `user` | The person (or application) sending input to the model. | "What pace should I target for a half marathon?" |
| `assistant` | The model's own prior responses, replayed to give conversational context. | The model's answer from the previous turn. |
| `tool` | Results returned from tool/function calls the model requested. | `{"temp": 18, "condition": "sunny"}` after the model asked to call `get_weather`. |

A multi-turn conversation replays the full history each time:

```python
messages = [
    {'role': 'system', 'content': 'You are a running coach.'},
    {'role': 'user', 'content': 'What pace should I target?'},
    {'role': 'assistant', 'content': 'Based on your recent training...'},
    {'role': 'user', 'content': 'What about for a half marathon?'},
]
```

#### Tool calling flow

The model never executes tools directly. It outputs structured JSON requesting a call, your code runs the function, and you pass the result back as a `tool` message:

1. **User asks**: "What's the weather in Paris?"
2. **Assistant responds** with `tool_calls`: `[{"function": {"name": "get_weather", "arguments": "{\"city\": \"Paris\"}"}}]`
3. **Your code** runs `get_weather("Paris")` and gets `{"temp": 18, "condition": "sunny"}`
4. **Tool message** sends the result back: `{'role': 'tool', 'content': '{"temp": 18, "condition": "sunny"}'}`
5. **Assistant** uses that result to write: "It's 18C and sunny in Paris."

This is how Claude Code, ChatGPT plugins, and any LLM-with-tools system works under the hood.

### Python SDKs for LLM APIs

Three packages for three providers, each with a different calling convention:

```python
# Ollama (local models, no API key, simplest API)
import ollama
response = ollama.chat('qwen3.5:35b', messages=[...])
answer = response['message']['content']

# OpenAI (needs OPENAI_API_KEY)
from openai import OpenAI
client = OpenAI()
response = client.chat.completions.create(model='gpt-4.1-mini', messages=[...])
answer = response.choices[0].message.content

# Anthropic (needs ANTHROPIC_API_KEY)
from anthropic import Anthropic
client = Anthropic()
response = client.messages.create(model='claude-sonnet-4-6', system='...', messages=[...])
answer = response.content[0].text
```

Key differences:
- **Ollama**: module-level functions (no client object), dict-style access (`response['message']`), system prompt goes in the messages list, model names use colons for size tags (`qwen3.5:35b`)
- **OpenAI**: client object, attribute access (`response.choices[0].message.content`), system prompt in messages list
- **Anthropic**: client object, attribute access, system prompt is a separate `system` parameter (not in messages)

Ollama can also be accessed via the OpenAI SDK by pointing at its OpenAI-compatible endpoint (`base_url="http://localhost:11434/v1"`). This is useful when you need MLflow's `openai.autolog()` integration.

### Ollama response structure

The full response from `ollama.chat()` contains more than just the answer:

| Field | Example | Meaning |
|-------|---------|---------|
| `model` | `'qwen3.5:35b'` | Which model ran |
| `done_reason` | `'stop'` | Stopped naturally vs. hit token limit |
| `total_duration` | `64475316917` (ns) | Total wall time (~64.5s) |
| `load_duration` | `9676765125` (ns) | Time loading model into GPU memory (~9.7s) |
| `prompt_eval_count` | `11` | Input tokens processed |
| `eval_count` | `1487` | Output tokens generated |
| `message.content` | `'Hello! How can I help?'` | The actual response |
| `message.thinking` | `'Thinking Process: ...'` | Chain-of-thought reasoning (Qwen3.5 has this on by default) |

### MLflow tracing for LLMs

MLflow offers two approaches for logging LLM calls:

1. **Autolog** (SDK-specific): `mlflow.openai.autolog()` or `mlflow.anthropic.autolog()` automatically traces every call through that SDK. No code changes needed. There is no `mlflow.ollama.autolog()`.

2. **Manual tracing**: `@mlflow.trace` decorator on any function. Works with any SDK, including Ollama. You control exactly what gets traced.

For local models via Ollama, `@mlflow.trace` is the way to go. If you want automatic tracing, route through the OpenAI SDK pointed at Ollama's compatible endpoint.

### MLflow's built-in LLM judges

MLflow provides pre-built LLM judges for evaluating model outputs. The **Correctness** judge (`mlflow.genai.judges.is_correct`) checks whether expected facts are supported by the response:

- **Prompt template**: `mlflow/genai/judges/prompts/correctness.py` defines the evaluation instructions
- **Judge function**: `is_correct(request, response, expected_facts)` returns a `Feedback` object with `"yes"` or `"no"`
- **Default model**: `gpt-4.1-mini` (non-Databricks) or Databricks' proprietary judge (Databricks environments)
- **Scorer wrapper**: `mlflow.genai.scorers.Correctness` wraps the judge for batch evaluation with `mlflow.evaluate()`

The judge checks if facts are **supported by** the response, not strict equivalence. The response can contain extra information and still be correct.

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

### Surrogate vs natural keys

**Strava's Activity ID ≠ the FIT filename.** The CSV `Filename` column (`activities/18666588095.fit.gz`) maps to `Activity ID` `17568798539` — they're different identifiers. The Activity ID is a surrogate key (auto-generated by Strava's database), while the filename is a natural key (derived from the data itself). Surrogate keys are preferred in data engineering because they're stable (don't change if data is re-uploaded), performant (integer joins beat string joins), and privacy-safe (can be obfuscated). When joining datasets from different sources, always verify the join key actually matches — an `INNER JOIN` returning 0 rows is the telltale sign of a key mismatch.

### Time-series regression & feature engineering

**Predicting pace at a given moment is a time-series regression problem.** The target (instantaneous speed) depends on current state (heart rate, elevation), recent history (HR trend over the last 60s, pace over the last 5 min), and contextual features (weather, total race distance). The core data engineering challenge is computing **lagged and rolling features** without data leakage: you can use past data (HR 30 seconds ago, rolling avg pace over last km) but never future data (avg HR for the rest of the run). `pandas.DataFrame.rolling()` computes windowed aggregates and `.shift()` creates lagged columns. Same pattern applies to stock price prediction, server latency forecasting, or any problem where the answer at time *t* depends on what happened before *t*.

### Confounding & omitted variable bias

**Humidity showed up as a top predictor of pace — but it's a confounder, not a cause.** The Napa Valley Marathon was early morning (cool, humid) with race shoes and full taper. Daily San Francisco training runs are late morning (dry, warm) at easy effort. The model sees high humidity → fast pace and learns a positive coefficient, but the true causal structure is `race_day → {humidity, shoes, effort, taper} → speed`. Without an `is_race` feature, the model distributes that effect across correlated proxies.

**Lasso vs Ridge expose confounding differently.** Pure Lasso (l1=1.0) made humidity the #1 feature — when forced to pick few features, it chose the single best proxy for "race day." Pure Ridge (l1=0.0) spread the signal across shoes, power, and humidity, dropping humidity to #8. This is because Ridge keeps all features and distributes weight, while Lasso concentrates on the most efficient proxies.

**This is omitted variable bias.** The fix is to add the missing variable (`is_race`) so the model can attribute the speed boost correctly. In experimentation (GrowthBook), randomization solves confounding by design. In observational ML, you have to engineer around it — either by adding the confounder as a feature, or using techniques like SHAP to decompose feature interactions.

### Target leakage

**Calories and power are downstream of speed, not causes of it.** Strava computes calories from speed, distance, and HR after the run. Running power (from the watch) is derived from speed + grade + weight in real time. Both are formula transformations of the target variable — including them inflates R² but doesn't add predictive value on genuinely unseen data. Removing them gives cleaner, more interpretable coefficients.

### LLM context windows and prompt caching

**LLMs are stateless.** The illusion of memory comes from replaying the full message history with every API call. As conversations grow, the message list grows linearly. Eventually it hits the model's **context window** (max tokens it can process at once). Management strategies: truncation (drop oldest messages), summarization (compress older turns into a summary), or a sliding window (keep only the last N turns).

**Prompt caching reduces cost without reducing context.** Providers (Anthropic, OpenAI) cache the repeated prefix of your messages. If turns 1-50 are identical to the last call and you only added turn 51, the cached prefix isn't reprocessed from scratch. Anthropic charges cached input tokens at ~10% of fresh input tokens. The model still "sees" everything, but compute cost and latency are reduced. Same idea as a database query cache: the full table scan only happens once.

### From linear baseline to XGBoost

**ElasticNet topped out at R² ≈ 0.40 on 100 runs.** With 10 runs (similar fitness, same routes), R² was ~0.77, but scaling to 100 runs spanning months of different fitness levels, terrains, and conditions exposed ElasticNet's limits. The problem is inherently non-linear: pace depends on *interactions* (elevation × fatigue, shoe × distance, HR × pct_complete) that a linear model can't capture.

**GBRT and XGBoost both capture non-linear interactions natively.** Each tree can split on elevation first, then split differently depending on fatigue level, without needing explicit interaction terms. On 75 runs: XGBoost hit R2 = 0.71 (RMSE 0.34), GBRT hit R2 = 0.67 (RMSE 0.36), ElasticNet hit R2 = 0.44 (RMSE 0.48). XGBoost's L2 regularization and early stopping gave it the edge over vanilla GBRT.

**The progression matters more than the final number.** Starting with ElasticNet gave us interpretable coefficients that surfaced the shoe effect, the humidity confounder, and the power leakage issue. None of those insights come from a tree model's feature importances. The right workflow is: interpretable baseline first (understand the data), then powerful model second (optimize predictions), tracked and compared in MLflow.

---

## Learning references

- "Hands-On Machine Learning" by Aurelien Geron

# MLflow End-to-End Implementation

A portfolio project demonstrating end-to-end ML experiment tracking and model management using MLflow, built around a real-world running pace prediction model using personal Strava data.

## Repository Structure

```
mlflow/
├── system_design/
│   └── mlflow_architecture.svg      # System design: tracking server, artifact store, model registry
├── strava_scripts/
│   ├── feature_engineering.py       # FIT file parsing, CSV merging, rolling/lagged feature engineering
│   ├── train_classical_ml.py        # ElasticNet training with MLflow experiment tracking
│   └── mini_script.py               # Exploratory FIT file parser
├── docker-compose/
│   ├── docker-compose.yml           # PostgreSQL + RustFS + MLflow server stack
│   ├── .env.dev.example             # Example environment variables
│   └── README.md                    # Docker setup guide
├── learnings.md                     # Notes taken while building the implementation
└── README.md
```

## System Design

![MLflow Architecture](system_design/mlflow_architecture.svg)

- **Tracking server**: experiment, run, and metric storage; local vs. remote backend
- **Artifact store**: model binaries, plots, feature importance charts, eval outputs
- **Model registry**: staging, production, and archived model versions with lineage
- **Serving layer**: MLflow model server, REST API, batch vs. real-time inference patterns

## Strava Pace Prediction (`strava_scripts/`)

Per-second running pace prediction using personal Strava data, tracked as MLflow experiments.

- **Data pipeline**: batch-parse Garmin FIT files (binary telemetry) and join with Strava CSV activity metadata via file ID bridge key
- **Target**: instantaneous speed (m/s) at each second of a run
- **Features**: elevation change (30s window), heart rate, elapsed time, % complete, shoe type (one-hot encoded), weather, `is_race` confounder control
- **Models**: ElasticNet as interpretable baseline (R² ≈ 0.40), XGBoost planned for non-linear interactions (R² target: 0.80+)
- **MLflow integration**: params, metrics (RMSE, R²), model artifact, and feature importance / residual plots logged per run

### Key findings

- **Elevation change** is the strongest predictor of instantaneous pace
- **Shoe type** has a measurable effect: carbon race shoes (METASPEED SKY) → faster, heavy trainers (GEL-NIMBUS 28) → slower
- **Humidity** appeared as a strong predictor but is a **confounder** — it proxies for race-day conditions (early morning, cool, tapered, race shoes). Adding `is_race` as a feature controls for this omitted variable bias
- **Calories and power** were identified as target leakage (both derived from speed) and removed

## Key Design Decisions

| Decision | Rationale |
|---|---|
| Local MLflow tracking server | No infra cost, reproducible across machines |
| Strava as training data | Real personal data, meaningful features, avoids synthetic data |
| Per-second granularity | Richer signal than activity-level aggregates, captures intra-run dynamics |
| ElasticNet first, then XGBoost | Linear baseline surfaces interpretable insights (shoe effects, confounders); tree model captures non-linear interactions |

## Requirements

```
mlflow
numpy
pandas
scikit-learn
matplotlib
fitparse
```

## Running the Project

```bash
# Start the Docker stack (PostgreSQL + RustFS + MLflow server)
cd docker-compose && docker compose up -d

# Train with default hyperparameters
python strava_scripts/train_classical_ml.py

# Train with custom hyperparameters
python strava_scripts/train_classical_ml.py --alpha 0.05 --l1-ratio 1.0 --limit 20

# Open MLflow UI to compare runs
open http://localhost:5000
```

## References

- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- Friedman et al., "Regularization Paths for Generalized Linear Models via Coordinate Descent" (2010)
- [Strava API Documentation](https://developers.strava.com)

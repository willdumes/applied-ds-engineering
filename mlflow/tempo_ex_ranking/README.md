# Tempo Exercise Ranker

Rank 600 exercises per user per day, show the top 10. Today this is done by manual reviewers when a user opens the app. Goal: replace the queue with a model.

## Approach

XGBRanker with `rank:pairwise`. Each user-day session is one ranking group. The model only ever sees pairs that share a `(user_id, day)`, so the loss is "did this completed item get a higher score than this skipped one in the same session?", not "what is P(complete)?".

Logistic regression as the baseline. Same features, same eval split, evaluated as a ranker by sorting `predict_proba` within each session. The baseline is the floor that justifies the ranker's existence.

Data is synthetic, generated from a latent process `z = fitness(user) - difficulty(exercise) + noise`. Three Iceberg tables (`tempo.users`, `tempo.exercises`, `tempo.events`) so every training run can be pinned to specific snapshot IDs and replayed bit-for-bit.

## Files

- `catalog.py`: PyIceberg catalog config, SQLite locally, REST-based in prod
- `generate_data.py`: synthesize users, exercises, events
- `feature_engineering.py`: read tables, encode features, build group sizes, NDCG and precision helpers
- `train_logistic.py`: baseline
- `train_xgboost.py`: ranker

## NDCG as the benchmark

Standard top-K ranking metric. Per session: sort items by predicted score, give each relevant item at position `r` a gain of `1 / log2(r+1)`, sum over the top K (the DCG), divide by the ideal DCG (best possible ordering). Average across sessions.

NDCG is bounded in [0, 1] but the lower bound is **not zero in practice**. For this setup (10 items per session, ~50% completion rate), random ordering already lands relevant items somewhere in the top 10:

```
IDCG (5 of 10 relevant) ≈ 2.95
E[DCG random]           ≈ 2.27
E[NDCG random]          ≈ 0.77
```

So 0.77 is the floor. The right way to read a model number is `(model - random) / (1 - random)`, the fraction of headroom captured.

## Results

| Model | NDCG@10 | Precision@3 | Precision@5 | Headroom captured |
|---|---|---|---|---|
| Random | 0.77 | ~0.50 | ~0.50 | 0% |
| Logistic regression | 0.800 | 0.574 | 0.571 | 13% |
| XGBRanker (pairwise) | 0.857 | 0.685 | 0.647 | 38% |

### Logistic baseline

`LogisticRegression(C=1.0, penalty='l2', solver='saga')` inside a `StandardScaler` pipeline. 12 features: 8 user numerics, `sex_M`, `zipcode_freq`, `bmi`, `exercise_id`. Predicts `P(complete)` per row, then sorts within each user-day session at eval time.

NDCG@10 = 0.800, capturing 13% of the gap from random to perfect. The near-flat precision profile (P@3 ≈ P@5) means the model has the gross order roughly right but does not separate well at the top of the list. That is the gap a ranker should close.

### XGBoost ranker

`XGBRanker(objective='rank:pairwise', n_estimators=300, learning_rate=0.05, max_depth=6, eval_metric='ndcg@10', early_stopping_rounds=20)`. Same 12 features, same time-based split (last 14 days held out). Group structure passed via `group_train` and `group_test` so XGBoost only forms pairs within sessions.

NDCG@10 = 0.857, best iteration 233 of 300 (early stopping triggered, healthy signal). Headroom captured jumps from 13% (logistic) to 38% (XGB).

The asymmetric lift is the headline:

- Precision@3: +0.111 over logistic
- Precision@5: +0.076 over logistic
- NDCG@10: +0.057 over logistic

The biggest gain is at the very top of the ranking. That is the signature of pairwise loss working as designed: gradient gets spent on getting the head of the list right, not on calibrating absolute probabilities across all 10 items. For a product that only shows the top 10, this is exactly the trade you want.

## Production considerations

The trained ranker is one piece of the serving stack, not the whole thing. Three issues worth flagging before shipping:

**Off-policy scoring.** Training only sees the 10 items the manual reviewers shipped. In production you score all 600 candidates per user per day. The model has never seen 590 of those (user, exercise) pairs, and tree models do not extrapolate gracefully. Standard fixes, in increasing order of effort: implicit negatives via random sampling of unshown items, inverse propensity scoring on the shown rows, or a two-stage retrieval + ranker architecture.

**Exploration.** Top-K from a single ranker creates a closed feedback loop: tomorrow's training data only contains items today's model already liked. Reserve 1 to 2 of the 10 slots per session for non-greedy picks. Side benefit: the random slots give an unbiased counterfactual for A/B analysis, which a pure exploit policy cannot.

**Diversity and recency.** A pure top-K is often 10 near-duplicates ("10 squat variants"). Filter items completed in the last 7 days, cap items per category or muscle group, then take top K from what is left. These are post-rank rules, not model concerns.

## How to run

From inside `mlflow/tempo_ex_ranking/`:

```bash
# 0. install deps if missing
pip install mlflow xgboost scikit-learn pyiceberg pyarrow pandas numpy

# 1. start the MLflow tracking server (separate terminal, leave running)
mlflow ui --host 127.0.0.1 --port 5000

# 2. generate the Iceberg tables (creates ./warehouse/ + catalog.db)
python generate_data.py

# 3. sanity-check the feature builder
python feature_engineering.py

# 4. train baseline, then ranker; both log to the same experiment
python train_logistic.py
python train_xgboost.py
```

Open `http://localhost:5000` and compare the runs side by side.

To replay a past run against the exact training data, pass the snapshot IDs that were logged to MLflow:

```bash
python train_xgboost.py \
  --users-snapshot 1771897722094072291 \
  --exercises-snapshot 2715693151168423041 \
  --events-snapshot 5287252656685166962
```

That is reproducibility from the data layer up, not from the model layer down.

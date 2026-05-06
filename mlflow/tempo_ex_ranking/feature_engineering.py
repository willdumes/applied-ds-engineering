"""Build a model-ready DataFrame from Iceberg tables.

Shared by train_xgboost.py (ranker) and train_logistic.py (baseline).
Reads the `tempo.users` / `tempo.exercises` / `tempo.events` Iceberg tables
through the project catalog. Each table read can be pinned to an explicit
snapshot_id so a training run can replay against the exact bytes it saw,
even after data has been regenerated.
"""
import numpy as np

from catalog import (
    EVENTS_TABLE, EXERCISES_TABLE, USERS_TABLE,
    get_catalog,
)

USER_NUMERIC = [
    'age', 'height', 'weight',
    'has_smart_watch', 'has_read_articles',
    'tempo_tenure', 'paying_customer', 'has_family_setup',
]


def _scan(table, snapshot_id=None):
    """Scan an Iceberg table at HEAD or a pinned snapshot, return pandas."""
    if snapshot_id is None:
        return table.scan().to_pandas()
    return table.scan(snapshot_id=snapshot_id).to_pandas()


def load_raw(users_snapshot=None, exercises_snapshot=None, events_snapshot=None):
    catalog = get_catalog()
    users_t = catalog.load_table(USERS_TABLE)
    exercises_t = catalog.load_table(EXERCISES_TABLE)
    events_t = catalog.load_table(EVENTS_TABLE)

    users = _scan(users_t, users_snapshot)
    exercises = _scan(exercises_t, exercises_snapshot)
    events = _scan(events_t, events_snapshot)

    snapshot_ids = {
        'users_snapshot_id': users_snapshot or users_t.current_snapshot().snapshot_id,
        'exercises_snapshot_id': exercises_snapshot or exercises_t.current_snapshot().snapshot_id,
        'events_snapshot_id': events_snapshot or events_t.current_snapshot().snapshot_id,
    }
    return users, exercises, events, snapshot_ids


def encode_users(users):
    """One-hot sex, frequency-encode zipcode, keep numerics."""
    out = users[['user_id'] + USER_NUMERIC].copy()
    out['sex_M'] = (users['sex'] == 'M').astype(int)

    # Frequency encoding for zipcode (high cardinality, no semantic order)
    zip_freq = users['zipcode'].value_counts(normalize=True)
    out['zipcode_freq'] = users['zipcode'].map(zip_freq).astype(float)

    # BMI as an engineered feature, since the latent process uses it
    out['bmi'] = users['weight'] / ((users['height'] / 100) ** 2)
    return out


def build_dataset(test_days=14, users_snapshot=None, exercises_snapshot=None, events_snapshot=None):
    """Join events with user features, split by day (last N days = test).

    Returns dict with X_train, y_train, group_train, X_test, y_test, group_test,
    test_keys, feature_cols, and snapshot_ids (the Iceberg snapshots actually
    read, for MLflow logging).
    """
    users, _, events, snapshot_ids = load_raw(
        users_snapshot=users_snapshot,
        exercises_snapshot=exercises_snapshot,
        events_snapshot=events_snapshot,
    )
    user_feats = encode_users(users)

    df = events.merge(user_feats, on='user_id', how='left')
    df = df.sort_values(['day', 'user_id']).reset_index(drop=True)

    max_day = df['day'].max()
    train_mask = df['day'] < (max_day - test_days + 1)
    train = df[train_mask].copy()
    test = df[~train_mask].copy()

    feature_cols = USER_NUMERIC + ['sex_M', 'zipcode_freq', 'bmi', 'exercise_id']

    def to_groups(part):
        # Group sizes per (user_id, day), preserving original row order.
        # XGBRanker expects a flat array of group lengths summing to len(part).
        sizes = part.groupby(['user_id', 'day'], sort=False).size().values
        return sizes

    out = {
        'X_train': train[feature_cols].astype(float),
        'y_train': train['completed'].astype(int).values,
        'group_train': to_groups(train),
        'X_test': test[feature_cols].astype(float),
        'y_test': test['completed'].astype(int).values,
        'group_test': to_groups(test),
        'test_keys': test[['user_id', 'day']].reset_index(drop=True),
        'feature_cols': feature_cols,
        'snapshot_ids': snapshot_ids,
    }

    print(f'Train: {len(train):,} rows, {len(out["group_train"]):,} sessions')
    print(f'Test:  {len(test):,} rows, {len(out["group_test"]):,} sessions')
    print(f'Train completion rate: {out["y_train"].mean():.3f}')
    print(f'Test completion rate:  {out["y_test"].mean():.3f}')
    print(f'Iceberg snapshots: {snapshot_ids}')
    return out


def ndcg_at_k(y_true, y_score, group_sizes, k=10):
    """Mean NDCG@k across sessions. Binary relevance (completed = 1)."""
    ndcgs = []
    start = 0
    for n in group_sizes:
        end = start + n
        labels = y_true[start:end]
        scores = y_score[start:end]
        if labels.sum() == 0:
            start = end
            continue
        order = np.argsort(-scores)
        gains = labels[order][:k]
        discounts = 1.0 / np.log2(np.arange(2, len(gains) + 2))
        dcg = (gains * discounts).sum()
        ideal = np.sort(labels)[::-1][:k]
        idcg = (ideal * discounts[:len(ideal)]).sum()
        ndcgs.append(dcg / idcg if idcg > 0 else 0.0)
        start = end
    return float(np.mean(ndcgs))


def precision_at_k(y_true, y_score, group_sizes, k=3):
    """Mean fraction of top-k predicted items that were completed."""
    precs = []
    start = 0
    for n in group_sizes:
        end = start + n
        labels = y_true[start:end]
        scores = y_score[start:end]
        order = np.argsort(-scores)
        top_k = labels[order][:k]
        precs.append(top_k.mean())
        start = end
    return float(np.mean(precs))


if __name__ == '__main__':
    data = build_dataset()
    print(f'Features: {data["feature_cols"]}')
    print(f'X_train shape: {data["X_train"].shape}')

"""Synthesize Tempo exercise-ranking data with a latent generative process.

Each user gets a latent fitness derived from real features so the model has
signal to recover. Each exercise gets a latent difficulty. Completion is
sampled from a logit of (fitness - difficulty + noise).

Writes three Iceberg tables (users, exercises, events). Each run produces
a new snapshot per table; old snapshots remain queryable for reproducible
model training (log the snapshot_id to MLflow alongside hyperparams).
"""
import argparse

import numpy as np
import pandas as pd
import pyarrow as pa

from catalog import (
    EVENTS_TABLE, EXERCISES_TABLE, USERS_TABLE,
    ensure_namespace, get_catalog,
)

N_USERS = 10_000
N_EXERCISES = 600
N_DAYS = 90
N_SHOWN_PER_DAY = 10


def generate_users(n_users, rng):
    zipcodes = [f'{z:05d}' for z in rng.integers(10_000, 99_999, 200)]
    return pd.DataFrame({
        'user_id': np.arange(n_users),
        'age': rng.integers(18, 80, n_users),
        'sex': rng.choice(['M', 'F'], n_users),
        'height': rng.normal(170, 10, n_users).clip(140, 210).round(1),
        'weight': rng.normal(75, 15, n_users).clip(45, 150).round(1),
        'has_smart_watch': rng.binomial(1, 0.4, n_users),
        'has_read_articles': rng.binomial(1, 0.3, n_users),
        'tempo_tenure': rng.exponential(180, n_users).astype(int),
        'zipcode': rng.choice(zipcodes, n_users),
        'paying_customer': rng.binomial(1, 0.6, n_users),
        'has_family_setup': rng.binomial(1, 0.2, n_users),
    })


def generate_exercises(n_exercises, rng):
    return pd.DataFrame({
        'exercise_id': np.arange(n_exercises),
        'difficulty': rng.normal(0, 1, n_exercises).round(3),
    })


def latent_fitness(users_df, rng):
    """Derive a latent fitness score from real user features.

    Coefficients are the ground truth the ranker should recover.
    """
    bmi = users_df['weight'] / ((users_df['height'] / 100) ** 2)
    return (
        0.02 * (40 - users_df['age'])
        + 0.30 * users_df['has_smart_watch']
        + 0.40 * users_df['has_read_articles']
        + 0.0015 * users_df['tempo_tenure']
        + 0.30 * users_df['paying_customer']
        + 0.25 * users_df['has_family_setup']
        - 0.05 * (bmi - 22).clip(-10, 20)
        + rng.normal(0, 0.4, len(users_df))
    ).values


def sample_shown_exercises(n_user_days, n_exercises, n_shown, rng, chunk=10_000):
    """For each user-day, pick n_shown exercise_ids without replacement.

    Chunked argpartition over random scores keeps memory bounded.
    """
    out = np.empty((n_user_days, n_shown), dtype=np.int32)
    for start in range(0, n_user_days, chunk):
        end = min(start + chunk, n_user_days)
        scores = rng.random((end - start, n_exercises))
        out[start:end] = np.argpartition(scores, n_shown, axis=1)[:, :n_shown]
    return out


def generate_events(users_df, exercises_df, n_days, n_shown, rng):
    n_users = len(users_df)
    n_exercises = len(exercises_df)
    n_user_days = n_users * n_days

    fitness = latent_fitness(users_df, rng)
    difficulty = exercises_df['difficulty'].values

    shown = sample_shown_exercises(n_user_days, n_exercises, n_shown, rng)
    exercise_ids = shown.flatten()

    user_ids = np.repeat(np.arange(n_users), n_days * n_shown)
    days = np.tile(np.repeat(np.arange(n_days), n_shown), n_users)

    z = fitness[user_ids] - difficulty[exercise_ids] + rng.normal(0, 0.7, len(exercise_ids))
    p = 1.0 / (1.0 + np.exp(-z))
    completed = (rng.random(len(p)) < p).astype(np.int8)

    return pd.DataFrame({
        'user_id': user_ids.astype(np.int32),
        'day': days.astype(np.int16),
        'exercise_id': exercise_ids.astype(np.int16),
        'completed': completed,
    })


def build_dataset(n_users, n_exercises, n_days, n_shown, seed):
    rng = np.random.default_rng(seed)
    users = generate_users(n_users, rng)
    exercises = generate_exercises(n_exercises, rng)
    events = generate_events(users, exercises, n_days, n_shown, rng)
    return users, exercises, events


def write_table(catalog, identifier, df):
    """Create-or-load an Iceberg table and overwrite it with df.

    overwrite() creates a new snapshot. Old snapshots remain queryable
    until snapshot expiration runs (a separate maintenance op).
    Returns the new snapshot_id.
    """
    arrow_table = pa.Table.from_pandas(df, preserve_index=False)
    try:
        table = catalog.load_table(identifier)
    except Exception:
        table = catalog.create_table(identifier, schema=arrow_table.schema)
    table.overwrite(arrow_table)
    return table.current_snapshot().snapshot_id


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate synthetic Tempo ranking data')
    parser.add_argument('--n-users', type=int, default=N_USERS)
    parser.add_argument('--n-exercises', type=int, default=N_EXERCISES)
    parser.add_argument('--n-days', type=int, default=N_DAYS)
    parser.add_argument('--n-shown', type=int, default=N_SHOWN_PER_DAY)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    users, exercises, events = build_dataset(
        args.n_users, args.n_exercises, args.n_days, args.n_shown, args.seed
    )

    catalog = get_catalog()
    ensure_namespace(catalog)

    users_snap = write_table(catalog, USERS_TABLE, users)
    exercises_snap = write_table(catalog, EXERCISES_TABLE, exercises)
    events_snap = write_table(catalog, EVENTS_TABLE, events)

    print(f'Users:     {users.shape[0]:>9,} rows  snapshot={users_snap}')
    print(f'Exercises: {exercises.shape[0]:>9,} rows  snapshot={exercises_snap}')
    print(f'Events:    {events.shape[0]:>9,} rows  snapshot={events_snap}')
    print(f'Completion rate: {events["completed"].mean():.3f}')
    print('Pin a model to these snapshots by passing --events-snapshot to the train scripts.')

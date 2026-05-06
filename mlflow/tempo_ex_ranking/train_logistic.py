"""Logistic regression baseline for the Tempo exercise ranker.

Predicts p(complete) per (user, exercise). Same features as the ranker.
We evaluate it as a ranker by sorting predictions inside each user-day
session, so MLflow comparisons against XGBRanker are apples-to-apples.
"""
import argparse

import numpy as np
import mlflow
import mlflow.sklearn

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from feature_engineering import build_dataset, ndcg_at_k, precision_at_k

MLFLOW_TRACKING_URI = 'http://localhost:5000'
EXPERIMENT_NAME = 'tempo_ex_ranking'


def train_and_log(data, C, penalty):
    X_train = data['X_train']
    y_train = data['y_train']
    X_test = data['X_test']
    y_test = data['y_test']
    group_test = data['group_test']

    # Scale + logistic regression. solver='saga' supports l1/l2/elasticnet.
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('lr', LogisticRegression(
            C=C,
            penalty=penalty,
            solver='saga',
            max_iter=2000,
            n_jobs=-1,
        )),
    ])
    pipe.fit(X_train, y_train)

    # Score the test set and rank within each session.
    y_score = pipe.predict_proba(X_test)[:, 1]
    ndcg10 = ndcg_at_k(y_test, y_score, group_test, k=10)
    prec3 = precision_at_k(y_test, y_score, group_test, k=3)
    prec5 = precision_at_k(y_test, y_score, group_test, k=5)

    mlflow.log_params({
        'model_type': 'logistic_regression',
        'C': C,
        'penalty': penalty,
        'solver': 'saga',
        'n_features': X_train.shape[1],
        **data['snapshot_ids'],
    })
    mlflow.log_metrics({
        'ndcg_at_10': ndcg10,
        'precision_at_3': prec3,
        'precision_at_5': prec5,
    })
    mlflow.sklearn.log_model(pipe, name='logreg_ranker')

    coefs = pipe.named_steps['lr'].coef_[0]
    idx = np.argsort(np.abs(coefs))[-10:][::-1]
    print('\nTop features by |coef|:')
    for feat, val in zip(np.array(data['feature_cols'])[idx], coefs[idx]):
        print(f'  {feat:20s}  {val:+.4f}')

    print(f'\nNDCG@10:     {ndcg10:.4f}')
    print(f'Precision@3: {prec3:.4f}')
    print(f'Precision@5: {prec5:.4f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Logistic regression baseline')
    parser.add_argument('--C', type=float, default=1.0)
    parser.add_argument('--penalty', type=str, default='l2', choices=['l1', 'l2'])
    parser.add_argument('--test-days', type=int, default=14)
    parser.add_argument('--users-snapshot', type=int, default=None)
    parser.add_argument('--exercises-snapshot', type=int, default=None)
    parser.add_argument('--events-snapshot', type=int, default=None)
    args = parser.parse_args()

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    data = build_dataset(
        test_days=args.test_days,
        users_snapshot=args.users_snapshot,
        exercises_snapshot=args.exercises_snapshot,
        events_snapshot=args.events_snapshot,
    )

    with mlflow.start_run(run_name=f'logreg_C{args.C}_{args.penalty}'):
        train_and_log(data, C=args.C, penalty=args.penalty)

    print(f'\nRun logged to {MLFLOW_TRACKING_URI}, experiment: {EXPERIMENT_NAME}')

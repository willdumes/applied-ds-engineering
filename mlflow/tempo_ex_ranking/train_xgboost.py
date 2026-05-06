"""Train an XGBRanker (pairwise) on the synthetic Tempo ranking data.

Each user-day session is one ranking group. The model learns to score
the 10 shown exercises so that completed ones rank above skipped ones.
"""
import argparse

import numpy as np
import mlflow
import xgboost as xgb

from feature_engineering import build_dataset, ndcg_at_k, precision_at_k

MLFLOW_TRACKING_URI = 'http://localhost:5000'
EXPERIMENT_NAME = 'tempo_ex_ranking'


def train_and_log(data, n_estimators, learning_rate, max_depth, reg_alpha, reg_lambda):
    # 1. Pull arrays out of the prepared data dict
    X_train = data['X_train']
    y_train = data['y_train']
    group_train = data['group_train']
    X_test = data['X_test']
    y_test = data['y_test']
    group_test = data['group_test']

    # 2. Define the ranker
    #    objective='rank:pairwise' optimizes pairs within each group
    #    (completed vs skipped within the same user-day session).
    model = xgb.XGBRanker(
        objective='rank:pairwise'
        , n_estimators=n_estimators
        , learning_rate=learning_rate
        , max_depth=max_depth
        , reg_alpha=reg_alpha
        , reg_lambda=reg_lambda
        , random_state=42
        , early_stopping_rounds=20
        , eval_metric='ndcg@10'
        , tree_method='hist'
    )

    # 3. Fit. group= tells XGBoost which rows belong to the same session.
    model.fit(
        X_train, y_train,
        group=group_train,
        eval_set=[(X_test, y_test)],
        eval_group=[group_test],
        verbose=False,
    )

    # 4. Score the test set and evaluate as a top-K ranker.
    y_score = model.predict(X_test)
    ndcg10 = ndcg_at_k(y_test, y_score, group_test, k=10)
    prec3 = precision_at_k(y_test, y_score, group_test, k=3)
    prec5 = precision_at_k(y_test, y_score, group_test, k=5)

    # 5. Log everything to MLflow.
    mlflow.log_params({
        'model_type': 'xgb_ranker',
        'objective': 'rank:pairwise',
        'n_estimators': n_estimators,
        'learning_rate': learning_rate,
        'max_depth': max_depth,
        'reg_alpha': reg_alpha,
        'reg_lambda': reg_lambda,
        'n_features': X_train.shape[1],
        'early_stopping_rounds': 20,
        **data['snapshot_ids'],
    })
    mlflow.log_metrics({
        'ndcg_at_10': ndcg10,
        'precision_at_3': prec3,
        'precision_at_5': prec5,
        'best_iteration': model.best_iteration,
    })
    mlflow.xgboost.log_model(model, name='xgb_ranker')

    # 6. Print top features for quick sanity.
    importances = model.feature_importances_
    idx = np.argsort(importances)[-10:][::-1]
    print('\nTop features by importance:')
    for feat, val in zip(np.array(data['feature_cols'])[idx], importances[idx]):
        print(f'  {feat:20s}  {val:.4f}')

    print(f'\nNDCG@10:        {ndcg10:.4f}')
    print(f'Precision@3:    {prec3:.4f}')
    print(f'Precision@5:    {prec5:.4f}')
    print(f'Best iteration: {model.best_iteration}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train XGBRanker on synthetic Tempo data')
    parser.add_argument('--n-estimators', type=int, default=300)
    parser.add_argument('--learning-rate', type=float, default=0.05)
    parser.add_argument('--max-depth', type=int, default=6)
    parser.add_argument('--reg-alpha', type=float, default=0.0)
    parser.add_argument('--reg-lambda', type=float, default=1.0)
    parser.add_argument('--test-days', type=int, default=14)
    parser.add_argument('--users-snapshot', type=int, default=None,
                        help='Pin users table to this Iceberg snapshot_id')
    parser.add_argument('--exercises-snapshot', type=int, default=None,
                        help='Pin exercises table to this Iceberg snapshot_id')
    parser.add_argument('--events-snapshot', type=int, default=None,
                        help='Pin events table to this Iceberg snapshot_id')
    args = parser.parse_args()

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    data = build_dataset(
        test_days=args.test_days,
        users_snapshot=args.users_snapshot,
        exercises_snapshot=args.exercises_snapshot,
        events_snapshot=args.events_snapshot,
    )

    with mlflow.start_run(
        run_name=f'xgb_ranker_n{args.n_estimators}_lr{args.learning_rate}_d{args.max_depth}'
    ):
        train_and_log(
            data,
            n_estimators=args.n_estimators,
            learning_rate=args.learning_rate,
            max_depth=args.max_depth,
            reg_alpha=args.reg_alpha,
            reg_lambda=args.reg_lambda,
        )

    print(f'\nRun logged to {MLFLOW_TRACKING_URI}, experiment: {EXPERIMENT_NAME}')

import argparse

import numpy as np
import mlflow

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

from feature_engineering import build_features

MLFLOW_TRACKING_URI = 'http://localhost:5000'
EXPERIMENT_NAME = 'strava'


def train_and_log(df, n_estimators, learning_rate, max_depth):
    X = df.drop(columns=['speed'])
    y = df['speed']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = GradientBoostingRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        random_state=42,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    mlflow.log_params(
        {
            'model_type': 'gbrt',
            'n_estimators': n_estimators,
            'learning_rate': learning_rate,
            'max_depth': max_depth,
            'n_features': X.shape[1],
        }
    )
    mlflow.log_metrics({'rmse': rmse, 'r2': r2})
    mlflow.sklearn.log_model(model, name='gbrt_pace_model')

    idx = np.argsort(model.feature_importances_)[-10:][::-1]
    top_vals = model.feature_importances_[idx]
    top_feat = X.columns[idx]
    print('\n Top 10 features by importance:')
    for feat, val in zip(top_feat, top_vals):
        print(f'  {feat:30s}  {val:.4f}')

    print('\nGBRT baseline run complete')
    print(f'RMSE: {rmse:.4f} m/s  |  R2: {r2:.4f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train GBRT on Strava running data')
    parser.add_argument('--n-estimators', type=int, default=200)
    parser.add_argument('--learning-rate', type=float, default=0.05)
    parser.add_argument('--max-depth', type=int, default=3)
    parser.add_argument('--limit', type=int, default=20)
    args = parser.parse_args()

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    df = build_features(limit=args.limit)

    with mlflow.start_run(
        run_name=f'gbrt_n{args.n_estimators}_lr{args.learning_rate}_d{args.max_depth}'
    ):
        train_and_log(
            df,
            n_estimators=args.n_estimators,
            learning_rate=args.learning_rate,
            max_depth=args.max_depth,
        )

    print(f'\nRun logged to {MLFLOW_TRACKING_URI}, experiment: {EXPERIMENT_NAME}')

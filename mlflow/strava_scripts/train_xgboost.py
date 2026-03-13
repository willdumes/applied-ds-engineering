import argparse

import numpy as np
import mlflow
import xgboost as xgb

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

from feature_engineering import build_features

MLFLOW_TRACKING_URI = 'http://localhost:5000'
EXPERIMENT_NAME = 'strava'


def train_and_log(df, n_estimators, learning_rate, max_depth, reg_alpha, reg_lambda):
    X = df.drop(columns=['speed'])
    y = df['speed']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = xgb.XGBRegressor(
        n_estimators=n_estimators
        , learning_rate=learning_rate
        , max_depth=max_depth
        , reg_alpha=reg_alpha
        , reg_lambda=reg_lambda
        , random_state=42
        , early_stopping_rounds=20
    )
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    mlflow.log_params(
        {
            'model_type': 'xgboost',
            'n_estimators': n_estimators,
            'learning_rate': learning_rate,
            'max_depth': max_depth,
            'reg_alpha': reg_alpha,
            'reg_lambda': reg_lambda,
            'n_features': X.shape[1],
            'early_stopping_rounds': 20,
        }
    )
    mlflow.log_metrics({'rmse': rmse, 'r2': r2})
    mlflow.xgboost.log_model(model, name='xgb_pace_model')

    idx = np.argsort(model.feature_importances_)[-10:][::-1]
    top_vals = model.feature_importances_[idx]
    top_feat = X.columns[idx]
    print('\n Top 10 features by importance:')
    for feat, val in zip(top_feat, top_vals):
        print(f'  {feat:30s}  {val:.4f}')

    print('\nXGBoost run complete')
    print(f'RMSE: {rmse:.4f} m/s  |  R2: {r2:.4f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train XGBoost on Strava running data')
    parser.add_argument('--n-estimators', type=int, default=300)
    parser.add_argument('--learning-rate', type=float, default=0.05)
    parser.add_argument('--max-depth', type=int, default=4)
    parser.add_argument('--reg-alpha', type=float, default=0.0)
    parser.add_argument('--reg-lambda', type=float, default=1.0)
    parser.add_argument('--limit', type=int, default=20)
    args = parser.parse_args()

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    df = build_features(limit=args.limit)

    with mlflow.start_run(
        run_name=f'xgb_n{args.n_estimators}_lr{args.learning_rate}_d{args.max_depth}'
    ):
        train_and_log(
            df,
            n_estimators=args.n_estimators,
            learning_rate=args.learning_rate,
            max_depth=args.max_depth,
            reg_alpha=args.reg_alpha,
            reg_lambda=args.reg_lambda,
        )

    print(f'\nRun logged to {MLFLOW_TRACKING_URI}, experiment: {EXPERIMENT_NAME}')

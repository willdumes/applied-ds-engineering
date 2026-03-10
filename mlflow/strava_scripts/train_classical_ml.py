import argparse

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mlflow

from sklearn.linear_model import ElasticNet
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

from feature_engineering import build_features

MLFLOW_TRACKING_URI = 'http://localhost:5000'
EXPERIMENT_NAME = 'strava'


def train_and_log(df, alpha, l1_ratio):
    """Train ElasticNet, log params/metrics/artifacts to MLflow, print top features."""
    X = df.drop(columns=['speed'])
    y = df['speed']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=10000, random_state=42)
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    coefs = pd.Series(model.coef_, index=X.columns)

    # Log to MLflow
    mlflow.log_params({'alpha': alpha, 'l1_ratio': l1_ratio, 'n_features': X.shape[1]})
    mlflow.log_metrics({'rmse': rmse, 'r2': r2, 'features_zeroed_by_l1': int((coefs.abs() < 1e-8).sum())})
    mlflow.sklearn.log_model(model, name='elasticnet_pace_model')

    # Print results
    print(f'\nElasticNet (alpha={alpha}, l1_ratio={l1_ratio})')
    print(f'RMSE: {rmse:.4f} m/s  |  R²: {r2:.4f}')
    top = coefs.abs().sort_values(ascending=False).head(10)
    print('\nTop 10 features:')
    for feat in top.index:
        print(f'  {"+" if coefs[feat] > 0 else "-"} {feat:30s}  {coefs[feat]:+.4f}')

    # Plot and log artifacts
    _plot_importance(coefs)
    _plot_residuals(y_test, y_pred)


def _plot_importance(coefs):
    top = coefs[coefs.abs().sort_values(ascending=False).head(10).index].sort_values()
    _, ax = plt.subplots(figsize=(10, 6))
    ax.barh(top.index, top.values, color=['#e74c3c' if v < 0 else '#2ecc71' for v in top.values])
    ax.set_xlabel('Coefficient (standardized)')
    ax.set_title('Top 10 Features Predicting Running Pace')
    ax.axvline(x=0, color='black', linewidth=0.5)
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=150)
    mlflow.log_artifact('feature_importance.png')


def _plot_residuals(y_test, y_pred):
    _, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].scatter(y_test, y_pred, alpha=0.1, s=5)
    axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    axes[0].set(xlabel='Actual speed (m/s)', ylabel='Predicted speed (m/s)', title='Predicted vs Actual')
    residuals = y_test - y_pred
    axes[1].hist(residuals, bins=50, edgecolor='black', alpha=0.7)
    axes[1].axvline(x=0, color='red', linestyle='--')
    axes[1].set(xlabel='Residual (m/s)', ylabel='Count', title=f'Residuals (mean={residuals.mean():.4f})')
    plt.tight_layout()
    plt.savefig('residuals.png', dpi=150)
    mlflow.log_artifact('residuals.png')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train ElasticNet on Strava running data')
    parser.add_argument('--alpha', type=float, default=0.01, help='Regularization strength')
    parser.add_argument('--l1-ratio', type=float, default=0.5, help='L1 vs L2 mix (0=Ridge, 1=Lasso)')
    parser.add_argument('--limit', type=int, default=10, help='Number of most recent FIT files to parse')
    args = parser.parse_args()

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    df = build_features(limit=args.limit)

    with mlflow.start_run(run_name=f'elasticnet_a{args.alpha}_l{args.l1_ratio}'):
        train_and_log(df, alpha=args.alpha, l1_ratio=args.l1_ratio)

    print(f'\nRun logged to {MLFLOW_TRACKING_URI}, experiment: {EXPERIMENT_NAME}')

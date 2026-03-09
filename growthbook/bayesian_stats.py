import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import norm, truncnorm
from sqlalchemy import create_engine


# connect to db
DB_URL = "postgresql://postgres:5432@localhost:5432/ds"
engine = create_engine(DB_URL)


def pull_logs(metric_col):
    """Query postgres and return a per-user aggregated DataFrame."""
    exp_df = pd.read_sql("SELECT * FROM exposures", engine)
    met_df = pd.read_sql("SELECT * FROM metrics", engine)

    first_exp = exp_df.groupby(['user_id', 'variation_id'])['timestamp'].min().reset_index()
    first_exp.columns = ['user_id', 'variation_id', 'first_exposed_at']

    merged_df = first_exp.merge(met_df, how='left', on=['user_id', 'variation_id'])
    merged_df = merged_df[merged_df['timestamp'] >= merged_df['first_exposed_at']]

    df = merged_df.groupby(['user_id', 'experiment_id', 'variation_id'])[metric_col].sum().reset_index()

    return df


def compute_summary_stats(df, metric_col):
    """Aggregate n, mean, variance per variation."""
    agg = df.groupby('variation_id')[metric_col].agg(['count', 'mean', 'var'])
    return {
        vid: {'n': row['count'], 'mean': row['mean'], 'variance': row['var']}
        for vid, row in agg.iterrows()
    }


def compute_relative_effect(ctrl, trt):
    """Point estimate + delta method variance for relative lift."""
    lift = (trt['mean'] - ctrl['mean']) / ctrl['mean']

    # Delta method: Var(M/D) with independent samples (cov = 0)
    # Term 1: treatment mean uncertainty scaled through the ratio
    # Term 2: control mean uncertainty, amplified by the denominator
    var_lift = (
        trt['variance'] / (trt['n'] * ctrl['mean'] ** 2)
        + ctrl['variance'] * trt['mean'] ** 2 / (ctrl['n'] * ctrl['mean'] ** 4)
    )
    return lift, var_lift


def bayesian_posterior(lift, var_lift):
    """Flat-prior Bayesian update — posterior IS the likelihood."""
    post_mean = lift
    post_std = np.sqrt(var_lift)

    # Chance to win: P(lift > 0)
    ctw = norm.sf(0, loc=post_mean, scale=post_std)

    # 95% credible interval
    ci = norm.ppf([0.025, 0.975], loc=post_mean, scale=post_std)

    # Risk: expected loss for choosing treatment vs control
    prob_ctrl_better = norm.cdf(0, loc=post_mean, scale=post_std)
    mean_if_negative = truncnorm.stats(-np.inf, (0 - post_mean) / post_std,
                                       loc=post_mean, scale=post_std, moments='m')
    mean_if_positive = truncnorm.stats((0 - post_mean) / post_std, np.inf,
                                       loc=post_mean, scale=post_std, moments='m')
    risk_trt = -float(prob_ctrl_better * mean_if_negative)
    risk_ctrl = float((1 - prob_ctrl_better) * mean_if_positive)

    return {
        'lift': lift,
        'chance_to_win': float(ctw),
        'ci_95': (float(ci[0]), float(ci[1])),
        'risk_treatment': risk_trt,
        'risk_control': risk_ctrl,
    }


def plot_posterior(result, var_lift):
    """Plot the posterior distribution of lift with 95% CI shaded."""
    post_mean = result['lift']
    post_std = np.sqrt(var_lift)
    ci_lo, ci_hi = result['ci_95']

    # x-axis: 4 standard deviations each side of the mean
    x = np.linspace(post_mean - 4 * post_std, post_mean + 4 * post_std, 300)
    y = norm.pdf(x, loc=post_mean, scale=post_std)

    fig, ax = plt.subplots(figsize=(10, 5))

    # Full curve
    ax.plot(x, y, color='steelblue', linewidth=2)

    # Shade the 95% credible interval
    ci_mask = (x >= ci_lo) & (x <= ci_hi)
    ax.fill_between(x[ci_mask], y[ci_mask], alpha=0.3, color='steelblue', label='95% CI')

    # Vertical lines: mean and zero
    ax.axvline(post_mean, color='steelblue', linestyle='--', linewidth=1,
               label=f'Lift = {post_mean:.2%}')
    ax.axvline(0, color='red', linestyle=':', linewidth=1, label='Zero effect')

    # Format x-axis as percentages
    ax.set_xlabel('Relative Lift')
    ax.set_ylabel('Density')
    ax.set_title('Posterior Distribution of Treatment Effect')
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:.1%}'))
    ax.legend()
    plt.tight_layout()
    plt.show()


def run():
    df = pull_logs('metric_0')
    stats = compute_summary_stats(df, 'metric_0')
    lift, var_lift = compute_relative_effect(stats['0'], stats['1'])
    result = bayesian_posterior(lift, var_lift)

    print(f"Control:   n={stats['0']['n']:.0f}  mean={stats['0']['mean']:.4f}")
    print(f"Treatment: n={stats['1']['n']:.0f}  mean={stats['1']['mean']:.4f}")
    print(f"\nLift: {result['lift']:.2%}")
    print(f"Chance to win: {result['chance_to_win']:.2%}")
    print(f"95% CI: ({result['ci_95'][0]:.2%}, {result['ci_95'][1]:.2%})")
    print(f"Risk (treatment): {result['risk_treatment']:.4%}")
    print(f"Risk (control):   {result['risk_control']:.4%}")

    plot_posterior(result, var_lift)


if __name__ == '__main__':
    run()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import t
from sqlalchemy import create_engine


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

    return merged_df.groupby(['user_id', 'experiment_id', 'variation_id'])[metric_col].sum().reset_index()


def compute_summary_stats(df, metric_col):
    """Aggregate count, mean, var per variation."""
    return df.groupby('variation_id')[metric_col].agg(['count', 'mean', 'var'])


def compute_relative_effect(ctrl, trt):
    """Point estimate + delta method variance for relative lift."""
    lift = (trt['mean'] - ctrl['mean']) / ctrl['mean']

    # Delta method: Var(M/D) with independent samples (cov = 0)
    var_lift = (
        trt['var'] / (trt['count'] * ctrl['mean'] ** 2)
        + ctrl['var'] * trt['mean'] ** 2 / (ctrl['count'] * ctrl['mean'] ** 4)
    )
    return lift, var_lift


def welch_test(lift, var_lift, ctrl, trt):
    """Welch's t-test — does NOT assume equal variance between groups."""
    se = np.sqrt(var_lift)
    t_stat = lift / se  # normalize to a standard t-distribution (0, 1)

    # Welch-Satterthwaite degrees of freedom
    se_c = ctrl['var'] / ctrl['count']
    se_t = trt['var'] / trt['count']
    df = (se_c + se_t) ** 2 / (se_c ** 2 / (ctrl['count'] - 1) + se_t ** 2 / (trt['count'] - 1))

    # Two-sided p-value (GrowthBook's exact formula)
    # "Under the null, what's the probability of being this surprised?"
    p_value = 2 * (1 - t.cdf(abs(t_stat), df))

    # 95% confidence interval
    halfwidth = t.ppf(0.975, df) * se
    ci = (lift - halfwidth, lift + halfwidth)

    return {
        'lift': lift,
        't_stat': t_stat,
        'df': df,
        'p_value': p_value,
        'ci_95': ci,
    }


def plot_null_vs_observed(result):
    """Plot the null t-distribution with the observed t-stat and p-value shaded."""
    x = np.linspace(-5, 5, 300)
    y = t.pdf(x, df=result['df'])

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(x, y, color='steelblue', linewidth=2)

    # Shade both tails beyond ±|t_stat| to show the p-value region
    t_abs = abs(result['t_stat'])
    left_tail = x <= -t_abs
    right_tail = x >= t_abs
    ax.fill_between(x[left_tail], y[left_tail], alpha=0.3, color='red', label=f'p-value = {result["p_value"]:.4f}')
    ax.fill_between(x[right_tail], y[right_tail], alpha=0.3, color='red')

    # Mark the observed t-stat
    ax.axvline(result['t_stat'], color='steelblue', linestyle='--', linewidth=1,
               label=f't-stat = {result["t_stat"]:.2f}')

    ax.set_xlabel('t')
    ax.set_ylabel('Density')
    ax.set_title('Null t-Distribution vs Observed Test Statistic')
    ax.legend()
    plt.tight_layout()
    plt.show()


def run():
    df = pull_logs('metric_0')

    stats = compute_summary_stats(df, 'metric_0')
    ctrl, trt = stats.loc['0'], stats.loc['1']

    lift, var_lift = compute_relative_effect(ctrl, trt)
    result = welch_test(lift, var_lift, ctrl, trt)

    print(f"Control:   n={ctrl['count']:.0f}  mean={ctrl['mean']:.4f}")
    print(f"Treatment: n={trt['count']:.0f}  mean={trt['mean']:.4f}")
    print(f"\nLift: {result['lift']:.2%}")
    print(f"t-stat: {result['t_stat']:.4f}")
    print(f"df: {result['df']:.1f}")
    print(f"p-value: {result['p_value']:.6f}")
    print(f"95% CI: ({result['ci_95'][0]:.2%}, {result['ci_95'][1]:.2%})")

    plot_null_vs_observed(result)


if __name__ == '__main__':
    run()

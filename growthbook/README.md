# GrowthBook End-to-End Implementation

A portfolio project demonstrating production-ready A/B testing infrastructure using GrowthBook, with both Bayesian and frequentist statistical approaches implemented from scratch in Python.

## Overview

This repo covers the full experimentation lifecycle: feature flagging, experiment assignment, results analysis, and statistical inference. It mirrors how modern experimentation platforms (GrowthBook, Eppo, Optimizely) operate under the hood, with explicit implementations of the two dominant statistical paradigms.

## Repository Structure

```
growthbook-implementation/
├── system_design/
│   └── schema.md              # System design: data model, event pipeline, assignment logic
├── bayesian_ab_testing.py     # Bayesian approach: Beta-Binomial, credible intervals, P(B > A)
├── frequentist_ab_testing.py  # Frequentist approach: t-test, z-test, p-values, power analysis
├── requirements.txt
└── README.md
```

## System Design

See `system_design/schema.md` for the full architecture, covering:

- **Experiment assignment**: user bucketing via deterministic hashing, traffic splitting, holdout groups
- **Event schema**: impression and conversion events, deduplication, session stitching
- **Data pipeline**: raw events to experiment results table (modeled in SQL/dbt style)
- **Feature flag lifecycle**: draft, running, stopped, and archived states

## Statistical Approaches

### Bayesian (`bayesian_ab_testing.py`)

GrowthBook defaults to Bayesian inference, which is now the industry standard in most modern platforms.

- **Prior**: Beta(1, 1) uninformative prior on conversion rate
- **Posterior update**: Beta(alpha + conversions, beta + non-conversions)
- **Output**: P(B > A) via Monte Carlo sampling, credible intervals, expected loss
- **Advantage over frequentist**: No peeking problem, results are interpretable at any point in time, supports dynamic traffic allocation (Thompson Sampling)

```python
# Example
posterior_a = beta(alpha=1 + conversions_a, beta=1 + (users_a - conversions_a))
posterior_b = beta(alpha=1 + conversions_b, beta=1 + (users_b - conversions_b))
p_b_beats_a = np.mean(posterior_b.rvs(100_000) > posterior_a.rvs(100_000))
```

### Frequentist (`frequentist_ab_testing.py`)

- **Two-proportion z-test** for conversion rate experiments
- **Welch's t-test** for continuous metrics (revenue, session length)
- **Power analysis**: sample size calculation via Cohen's h (proportions) and Cohen's d (means)
- **Multiple testing correction**: Bonferroni and Benjamini-Hochberg (FDR)

```python
# Example
from scipy import stats
z_stat, p_value = stats.proportions_ztest([conv_a, conv_b], [n_a, n_b])
```

## Key Design Decisions

| Decision | Rationale |
|---|---|
| Bayesian as primary | Matches GrowthBook default, more intuitive for stakeholders |
| Beta-Binomial conjugate | Closed-form posterior, no MCMC needed for conversion experiments |
| Welch's t-test over Student's | Does not assume equal variance across variants |
| Sequential-aware Bayesian | Supports early stopping without inflating Type I error |

## Requirements

```
numpy
scipy
pandas
matplotlib
pymc  # optional, for more complex models
```

## References

- [GrowthBook Statistics Documentation](https://docs.growthbook.io/statistics)
- Gelman et al., *Bayesian Data Analysis* (3rd ed.)
- Kohavi et al., *Trustworthy Online Controlled Experiments*

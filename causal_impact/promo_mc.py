"""Monte Carlo simulation of a 40-year career path across levels L1..L12.

Starts at L3. Promotion probability per year depends on current level:
    L3, L4, L5   -> 0.50
    L6, L7, L8   -> 0.25
    L9, L10, L11 -> 0.01
    L12          -> terminal
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator
from scipy.stats import binom

# Simulation knobs: 1M independent careers, each tracked over 40 years.
# SEED makes the RNG deterministic so re-runs produce identical paths.
N_PATHS = 1_000_000
YEARS = 40
START_LEVEL = 3
MAX_LEVEL = 12
SEED = 42

# Promotion probability lookup, indexed by level number.
# Indices 0..2 are padding so PROMO_PROB[3] corresponds to L3 (no arithmetic offset needed).
PROMO_PROB = np.array([
    0.00, 0.00, 0.00,    # L0, L1, L2  (unused, kept for index alignment)
    0.50, 0.50, 0.50,    # L3, L4, L5
    0.25, 0.25, 0.25,    # L6, L7, L8
    0.01, 0.01, 0.01,    # L9, L10, L11
    0.00,                # L12 terminal (0 prob -> stays there forever)
])


def promo_prob(level: np.ndarray) -> np.ndarray:
    """Return per-path yearly promotion probability for the given levels."""
    # Fancy indexing: if `level` is an array of shape (N,), the result is also (N,),
    # with each entry looked up from PROMO_PROB. This is the vectorized equivalent
    # of `[PROMO_PROB[l] for l in level]` but runs in C.
    return PROMO_PROB[level]


def simulate() -> np.ndarray:
    """Run N_PATHS careers for YEARS years and return a (N_PATHS, YEARS+1) array."""
    rng = np.random.default_rng(SEED)

    # Pre-allocate the full trajectory grid and seed column 0 with START_LEVEL.
    # Shape (N_PATHS, YEARS+1): each row is one career, each column is a year snapshot.
    levels = np.full((N_PATHS, YEARS + 1), START_LEVEL, dtype=int)

    # March forward year by year. Vectorized across all N_PATHS paths simultaneously.
    for t in range(1, YEARS + 1):
        current = levels[:, t - 1]                            # levels at start of year t
        # Draw a Bernoulli (n=1) for each path with its level-dependent probability.
        # `promoted` is a 0/1 array: 1 means the path got promoted this year.
        promoted = binom.rvs(n=1, p=promo_prob(current), random_state=rng)
        # Advance by 0 or 1, then cap at MAX_LEVEL so L12 stays terminal.
        levels[:, t] = np.minimum(current + promoted, MAX_LEVEL)

    return levels


def plot(levels: np.ndarray) -> None:
    # x-axis: integer years 0..YEARS (the original sampling grid from simulate()).
    years_axis = np.arange(YEARS + 1)

    # Compute 1st, 50th, and 99th percentile of the level across all paths, per year.
    # `axis=0` collapses the N_PATHS dimension, leaving one value per year.
    p01 = np.percentile(levels, 1, axis=0)
    p50 = np.percentile(levels, 50, axis=0)
    p99 = np.percentile(levels, 99, axis=0)

    # Smooth the step-function percentiles by resampling onto 500 points and
    # interpolating with PCHIP (monotone-preserving cubic). PCHIP avoids the
    # overshoot a natural cubic spline would introduce at integer jumps.
    x_smooth = np.linspace(0, YEARS, 500)
    p01_s = PchipInterpolator(years_axis, p01)(x_smooth)
    p50_s = PchipInterpolator(years_axis, p50)(x_smooth)
    p99_s = PchipInterpolator(years_axis, p99)(x_smooth)

    _, ax = plt.subplots(figsize=(10, 6))
    # Grey band: where 98% of simulated careers fall at each year.
    ax.fill_between(x_smooth, p01_s, p99_s, color="grey", alpha=0.35, label="1st to 99th pct")
    # Black line: the "typical" career trajectory.
    ax.plot(x_smooth, p50_s, color="black", linewidth=2.5, label="Median")

    # Y axis as discrete level labels L1..L12 rather than raw numbers.
    ax.set_xlabel("Years into career")
    ax.set_ylabel("Level")
    ax.set_yticks(range(1, MAX_LEVEL + 1))
    ax.set_yticklabels([f"L{i}" for i in range(1, MAX_LEVEL + 1)])
    ax.set_xlim(0, YEARS)
    ax.set_ylim(1, MAX_LEVEL)
    ax.set_title(f"Career level over {YEARS} years ({N_PATHS:,} Monte Carlo paths)")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("promo_mc.png", dpi=120)
    plt.show()


if __name__ == "__main__":
    levels = simulate()
    plot(levels)

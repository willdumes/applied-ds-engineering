"""Synthetic dataset for the CausalImpact notebook.

Mimics the original Kaggle ``example.csv`` shape: 45 European cities by 127 daily
observations, with Barcelona receiving a treatment effect from 2022-07-01.
The pre-period spans 2022-03-10 to 2022-06-30; the post-period 2022-07-01 to 2022-07-13.

Run this once before opening ``causal-impact-framework.ipynb``::

    python generate_example_data.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

CITIES = [
    "Barcelona", "London", "Paris", "Berlin", "Rome",
    "Madrid", "Moscow", "Istanbul", "Kiev", "Vienna",
    "Hamburg", "Munich", "Naples", "Athens", "Warsaw",
    "Bucharest", "Budapest", "Sofia", "Brussels", "Amsterdam",
    "Prague", "Birmingham", "Stockholm", "Minsk", "Oslo",
    "Copenhagen", "Helsinki", "Dublin", "Lyon", "Marseille",
    "Lille", "Geneva", "Zurich", "Milan", "Florence",
    "Nice", "Lisbon", "Edinburgh", "Porto", "Glasgow",
    "Reykjavik", "Belgrade", "Krakow", "Seville", "Turin",
]

CORRELATED_WITH_BARCELONA = {
    "Paris", "Kiev", "Warsaw", "Birmingham", "Stockholm",
    "Minsk", "Oslo", "Copenhagen", "Lyon", "Lille", "Belgrade", "Turin",
}


def generate(seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    dates = pd.date_range("2022-03-10", "2022-07-14", freq="D")
    n = len(dates)
    treated = dates >= pd.Timestamp("2022-07-01")

    factor = rng.normal(0, 1, n).cumsum() / np.sqrt(n)
    dow = np.array([d.weekday() for d in dates])
    weekly = -8000 * (dow >= 5)

    df = pd.DataFrame(index=dates)
    df.index.name = "Date"

    barcelona = 180_000 + 30_000 * factor + weekly + rng.normal(0, 6000, n)
    barcelona[treated] += 220_000
    df["Barcelona"] = barcelona

    for city in CITIES[1:]:
        base = float(rng.uniform(30_000, 600_000))
        sd = base * 0.08
        if city in CORRELATED_WITH_BARCELONA:
            series = base + (base / 10) * factor + rng.normal(0, sd, n)
        else:
            series = base + rng.normal(0, sd, n)
        df[city] = series

    df["The Hague"] = np.nan
    return df


def main() -> None:
    out = Path(__file__).parent / "data" / "example.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df = generate()
    df.to_csv(out)
    print(f"Wrote {df.shape[0]} rows by {df.shape[1]} cols to {out}")


if __name__ == "__main__":
    main()

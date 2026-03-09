import argparse
import pandas as pd
import psycopg2
import numpy as np
from sqlalchemy import create_engine

DB_URL = "postgresql://postgres:5432@localhost:5432/ds"


def run(metric_name, mean_t, mean_c, std):
    engine = create_engine(DB_URL)

    # Query the exposures into a DataFrame (psycopg2).
    df = pd.read_sql("SELECT DISTINCT user_id, experiment_id, variation_id FROM exposures", engine)

    # Apply a normal distribution to a new column to generate a metric for treatment/control
    df[metric_name] = np.where(df['variation_id'] == "1", np.random.normal(mean_t, std, len(df)), np.random.normal(mean_c, std, len(df)))
    df["timestamp"] = pd.Timestamp.now(tz="UTC")
 
    # Write the DataFrame to a database
    df.to_sql("metrics", engine, if_exists="append", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--metric_name", default="metric_0")
    parser.add_argument("--mean_t", type=float, default=11)
    parser.add_argument("--mean_c", type=float, default=10)
    parser.add_argument("--std", type=float, default=3)
    args = parser.parse_args()
    run(args.metric_name, args.mean_t, args.mean_c, args.std)
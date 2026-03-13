from fitparse import FitFile
import gzip
import os
import pandas as pd

ACTIVITIES_DIR = '/Users/willd/Development/applied-ds-engineering/mlflow/strava_data/activities'
CSV_PATH = '/Users/willd/Development/applied-ds-engineering/mlflow/strava_data/activities.csv'


def extract_all_fit_records(activities_dir, limit=None):
    """Loop through .fit and .fit.gz files, parse records, return combined DataFrame.
    If limit is set, only parse the N most recent files (highest numeric ID = most recent).
    """
    all_dfs = []
    skipped = 0

    files = sorted(os.listdir(activities_dir))
    fit_files = [f for f in files if f.endswith('.fit') or f.endswith('.fit.gz')]

    if limit:
        fit_files = sorted(fit_files, key=lambda f: int(f.split('.')[0]), reverse=True)[:limit]

    for filename in fit_files:
        activity_id = filename.split('.')[0]
        filepath = os.path.join(activities_dir, filename)

        try:
            if filename.endswith('.fit.gz'):
                with gzip.open(filepath, 'rb') as gz:
                    data = gz.read()
                fit = FitFile(data)
            else:
                fit = FitFile(filepath)

            records = [{field.name: field.value for field in msg.fields}
                       for msg in fit.get_messages('record')]

            if records:
                df = pd.DataFrame(records)
                df['file_id'] = activity_id
                all_dfs.append(df)
        except Exception:
            skipped += 1

    print(f'Parsed {len(all_dfs)} FIT files, skipped {skipped} (parse errors)')
    return pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()


def extract_activities_from_csv(csv_path):
    df = pd.read_csv(csv_path)
    runs = df[df['Activity Type'] == 'Run'][[
        'Activity ID', 'Activity Date', 'Activity Name', 'Activity Type',
        'Elapsed Time', 'Distance', 'Moving Time', 'Elevation Gain',
        'Average Heart Rate', 'Max Heart Rate', 'Average Cadence',
        'Calories', 'Activity Gear',
        'Weather Temperature', 'Humidity', 'Filename',
    ]].copy()
    # Extract the numeric file ID from Filename (e.g., "activities/18666588095.fit.gz" → "18666588095")
    runs['file_id'] = runs['Filename'].str.extract(r'activities/(\d+)')

    # Flag race activities to control for confounding (race day → shoes + weather + effort → speed)
    runs['is_race'] = runs['Activity Name'].str.contains('Marathon|Race|race', na=False).astype(int)

    return runs


def merge_activities_and_records(activities_df, records_df):
    return activities_df.merge(records_df, how='inner', on='file_id')


def add_engineered_features(df):
    """Add lagged, rolling, and derived features per activity.

    All rolling/lag features use only past data (no leakage).
    Target variable: speed (m/s).
    """
    feature_dfs = []

    for file_id, group in df.groupby('file_id'):
        g = group.sort_values('timestamp').copy()

        # --- Elapsed seconds since start of this activity ---
        g['elapsed_sec'] = (g['timestamp'] - g['timestamp'].iloc[0]).dt.total_seconds()

        # --- Fraction of run completed (by distance) ---
        max_dist = g['distance'].max()
        if max_dist and max_dist > 0:
            g['pct_complete'] = g['distance'] / max_dist
        else:
            g['pct_complete'] = 0.0

        # --- Elevation change over last 30s ---
        g['elevation_change_30s'] = g['enhanced_altitude'] - g['enhanced_altitude'].shift(30)

        # --- Rolling power (effort proxy) ---
        g['power_rolling_60s'] = g['power'].rolling(window=60, min_periods=10).mean()

        # --- Heart rate rolling (when available) ---
        g['hr_rolling_60s'] = g['heart_rate'].rolling(window=60, min_periods=10).mean()

        feature_dfs.append(g)

    return pd.concat(feature_dfs, ignore_index=True)


def prepare_model_input(df):
    """Select features and target, one-hot encode categoricals, drop rows with NaN target."""
    target = 'speed'

    # Drop key/identifier columns and redundant columns
    drop_cols = [
        'Activity ID', 'file_id', 'Activity Date', 'Activity Name',
        'Activity Type', 'Filename', 'timestamp', 'enhanced_speed',
        'position_lat', 'position_long', 'gps_accuracy',
        'Distance', 'Elapsed Time', 'Moving Time',
    ]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # One-hot encode Activity Gear (shoes)
    if 'Activity Gear' in df.columns:
        df = pd.get_dummies(df, columns=['Activity Gear'], prefix='shoe', dtype=int)

    # Drop rows where target is missing
    df = df.dropna(subset=[target])

    # Drop rows where speed is near-zero (standing still, GPS warmup)
    df = df[df[target] > 0.5]

    # Fill remaining NaNs with 0 (missing HR, power in some records)
    df = df.fillna(0)

    return df


def build_features(limit=10):
    """End-to-end pipeline: parse → merge → engineer → prepare."""
    records_df = extract_all_fit_records(ACTIVITIES_DIR, limit=limit)
    activities_df = extract_activities_from_csv(CSV_PATH)
    merged_df = merge_activities_and_records(activities_df, records_df)
    featured_df = add_engineered_features(merged_df)
    model_df = prepare_model_input(featured_df)

    print(f'Final dataset: {model_df.shape[0]} rows, {model_df.shape[1]} columns')
    print(f'Target (speed) range: {model_df["speed"].min():.2f} - {model_df["speed"].max():.2f} m/s')
    print(f'Features: {[c for c in model_df.columns if c != "speed"]}')

    return model_df


if __name__ == "__main__":
    df = build_features(limit=10)
    print(f'\nSample row:')
    print(df.iloc[100].to_string())
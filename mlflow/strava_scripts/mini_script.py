from fitparse import FitFile
import pandas as pd

f = FitFile('/Users/willd/Development/applied-ds-engineering/mlflow/strava_data/activities/18666588095.fit')

def record_to_dict(record):
    return {field.name: field.value for field in record.fields}

records = [record_to_dict(r) for r in f.get_messages('record')]
df = pd.DataFrame(records)

print(f'Shape: {df.shape}')
print(f'\nColumns: {list(df.columns)}')
print(f'\nFirst 5 rows:')
print(df.head().to_string())
print(f'\nBasic stats:')
print(df.describe().to_string())
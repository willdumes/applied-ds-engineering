import sys
from pathlib import Path

from fitparse import FitFile
import pandas as pd

_DEFAULT_DIR = Path(__file__).resolve().parent.parent / 'strava_data' / 'activities'

if len(sys.argv) > 1:
    fit_path = Path(sys.argv[1])
else:
    candidates = sorted(_DEFAULT_DIR.glob('*.fit'))
    if not candidates:
        sys.exit(f'No .fit files found in {_DEFAULT_DIR}. Pass a path: python mini_script.py /path/to/activity.fit')
    fit_path = candidates[0]

f = FitFile(str(fit_path))

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
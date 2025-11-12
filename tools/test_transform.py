from pathlib import Path
import joblib
import pandas as pd

p = Path(__file__).parents[1] / 'models' / 'preprocessor.pkl'
print('preprocessor path:', p)
pre = joblib.load(p)
print('Preprocessor type:', type(pre))

# Load processed data and take one sample, set month to 'jan'
data_path = Path('data/processed/bank-processed.csv')
df = pd.read_csv(data_path)
print('processed data shape:', df.shape)
row = df.iloc[[0]].copy()
if 'month' in row.columns:
    row.at[row.index[0], 'month'] = 'jan'
    print('Set month to jan for test')
else:
    print('month column not found in processed data')

# Attempt to transform
try:
    out = pre.transform(row)
    print('Transform succeeded. Output shape:', out.shape)
    print(out.head())
except Exception as e:
    print('Transform failed with exception:')
    raise

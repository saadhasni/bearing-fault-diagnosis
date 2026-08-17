"""Entry point. Run from the project root:
    python src/main.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from data import FS, build_dataset
from experiment import build_feature_table, compare, within_load_baseline
from features import TIME_KEYS, FREQ_KEYS

if __name__ == '__main__':
    Path('results').mkdir(exist_ok=True)

    print('Loading recordings...')
    X, y, load, rpm = build_dataset(data_dir='data')
    print(f'\n{len(X)} windows total')
    print(pd.crosstab(y, load).to_string(), '\n')

    print('Extracting features...')
    df = build_feature_table(X, y, load, rpm, FS)
    df.to_csv('results/features.csv', index=False)
    print(f'  saved results/features.csv  ({df.shape[0]} x {df.shape[1]})\n')

    print('=' * 62)
    print('FLAWED PROTOCOL (random split, load ignored)')
    print('=' * 62)
    for name, keys in [('time only', TIME_KEYS), ('frequency only', FREQ_KEYS),
                       ('both', TIME_KEYS + FREQ_KEYS)]:
        print(f'  {name:16s} {within_load_baseline(df, keys):.3f}')

    loads = sorted(df.load.unique())
    if len(loads) < 2:
        print('\n' + '!' * 62)
        print(f'ONLY ONE MOTOR LOAD FOUND: {loads[0]}HP')
        print('The cross-load experiment needs at least two loads.')
        print('Download the 1, 2 and 3 HP files listed in data.py and re-run.')
        print('!' * 62)
        raise SystemExit(0)

    print('\n' + '=' * 62)
    print('CORRECTED PROTOCOL (leave-one-load-out)')
    print('=' * 62)
    summary, results = compare(df, model='rf')
    print(summary.to_string(index=False))
    summary.to_csv('results/summary.csv', index=False)

    for name, r in results.items():
        print(f'\n--- {name}  (mean {r["mean"]:.3f} +/- {r["std"]:.3f}) ---')
        print(pd.DataFrame(r['confusion'], index=r['labels'],
                           columns=r['labels']).to_string())

    imp = results['both']['folds'][0]['importances']
    print('\nTop 8 features (both, fold 0):')
    for k, v in sorted(imp.items(), key=lambda x: -x[1])[:8]:
        print(f'  {k:18s} {v:.3f}')

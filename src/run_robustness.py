"""Run the robustness sweeps. From the project root:
    python src/run_robustness.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
from data import FILES, RPM_BY_LOAD, load_de_signal, build_dataset
from robustness import noise_sweep, window_sweep

if __name__ == '__main__':
    Path('results').mkdir(exist_ok=True)

    print('Loading...')
    X, y, load, rpm = build_dataset(data_dir='data', verbose=False)
    print(f'  {len(X)} windows, loads {sorted(set(load))}\n')

    print('NOISE SWEEP (train clean, test noisy)')
    ns = noise_sweep(X, y, load, rpm)
    ns.to_csv('results/noise_sweep.csv', index=False)

    print('\nWINDOW LENGTH SWEEP')
    sigs = []
    for lab, by_load in FILES.items():
        for hp, fn in by_load.items():
            p = Path('data') / fn
            if p.exists():
                sigs.append((load_de_signal(p), lab, hp, RPM_BY_LOAD[hp]))
    ws = window_sweep(sigs)
    ws.to_csv('results/window_sweep.csv', index=False)
    print('\nSaved results/noise_sweep.csv and results/window_sweep.csv')

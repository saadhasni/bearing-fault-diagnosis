"""CWRU bearing dataset loader - CROSS-LOAD protocol.

Download: https://engineering.case.edu/bearingdatacenter/download-data-file
Free, no registration. You need 16 files: 4 fault classes x 4 motor loads.

WHY CROSS-LOAD
--------------
A first attempt using one recording per class scored 100% on every feature set,
including time-domain statistics alone. That result was an artefact: with a
single recording per class, every window of a class shares the same gain and
noise floor, so a classifier can identify the RECORDING rather than the FAULT.
Feature importances confirmed it - rms, std and peak dominated, not the defect
frequencies.

The fix is to train on some motor loads and test on a load never seen in
training. Vibration amplitude changes with load, so recording-level shortcuts
stop transferring. Envelope-spectrum features should survive, because the
defect frequencies are recomputed from the shaft speed of each condition.

This is the standard rigorous protocol for CWRU and it is what turns the
project from a classification demo into an actual experiment.
"""

from pathlib import Path

import numpy as np
from scipy.io import loadmat

FS = 12000                       # sampling rate, Hz (12k drive-end files)

# Motor load (HP) -> approximate shaft speed (rpm), from the CWRU site
RPM_BY_LOAD = {0: 1797, 1: 1772, 2: 1750, 3: 1730}

# label -> {load: filename}.  0.007" fault diameter, 12k drive end.
# Outer race files are the CENTRED (@6:00) variant.
FILES = {
    'normal':     {0: '97.mat',  1: '98.mat',  2: '99.mat',  3: '100.mat'},
    'inner_race': {0: '105.mat', 1: '106.mat', 2: '107.mat', 3: '108.mat'},
    'ball':       {0: '118.mat', 1: '119.mat', 2: '120.mat', 3: '121.mat'},
    'outer_race': {0: '130.mat', 1: '131.mat', 2: '132.mat', 3: '133.mat'},
}


def load_de_signal(path):
    """Extract the drive-end accelerometer channel from a CWRU .mat file."""
    mat = loadmat(path)
    keys = [k for k in mat if k.endswith('_DE_time')]
    if not keys:
        raise KeyError(f"no *_DE_time variable in {path}; found {list(mat)}")
    return np.asarray(mat[keys[0]]).ravel().astype(np.float64)


def window(x, size=2048, overlap=0.5):
    """Split a recording into fixed-length windows.

    2048 samples at 12 kHz is ~171 ms, about 5 shaft revolutions at 1797 rpm -
    enough for several defect impacts per window.
    """
    step = int(size * (1 - overlap))
    return np.array([x[i:i + size] for i in range(0, len(x) - size + 1, step)])


def build_dataset(data_dir='data', size=2048, overlap=0.5, files=None,
                  verbose=True):
    """Load every (class, load) recording into windows.

    Returns X (n, size), y labels (n,), load (n,), rpm (n,).
    The train/test split is decided later by LOAD, not stored here.
    """
    files = files or FILES
    data_dir = Path(data_dir)
    X, y, load, rpm = [], [], [], []
    missing = []
    for label, by_load in sorted(files.items()):
        for hp, fname in sorted(by_load.items()):
            p = data_dir / fname
            if not p.exists():
                missing.append(f"{label} {hp}HP -> {fname}")
                continue
            w = window(load_de_signal(p), size, overlap)
            X.append(w)
            y.append(np.full(len(w), label))
            load.append(np.full(len(w), hp, dtype=int))
            rpm.append(np.full(len(w), RPM_BY_LOAD[hp], dtype=int))
            if verbose:
                print(f"  {label:11s} {hp}HP  {fname:9s} {len(w):4d} windows")
    if missing:
        print("\nMISSING FILES (skipped):")
        for m in missing:
            print("  ", m)
        print()
    if not X:
        raise FileNotFoundError("no data files found in " + str(data_dir))
    return (np.concatenate(X), np.concatenate(y),
            np.concatenate(load), np.concatenate(rpm))

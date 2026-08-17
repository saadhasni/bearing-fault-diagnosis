"""Robustness experiments.

MOTIVATION
----------
Under the leave-one-load-out protocol every feature set scored 1.000. That is
not a bug: CWRU 0.007" seeded faults produce very large class margins (outer
race kurtosis ~7.6 against ~2.8 for normal), and near-perfect accuracy on this
subset is routinely reported in the literature.

An accuracy table of all 1.000s answers nothing. These two sweeps degrade the
problem in physically meaningful ways and ask where each feature set BREAKS -
which is the informative question.

  1. NOISE     Add white Gaussian noise at decreasing SNR. Real installations
               have sensor noise, electrical interference and vibration from
               neighbouring machinery. Which features survive contamination?

  2. WINDOW    Shorten the analysis window. Short windows mean lower detection
               latency and cheaper hardware, but fewer defect impacts per
               decision. What is the minimum usable window?

Both produce CURVES rather than single numbers, which is a far better result.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

from data import FS
from features import extract, TIME_KEYS, FREQ_KEYS


def add_noise(x, snr_db, rng):
    """Add white Gaussian noise at a specified SNR, in dB."""
    p_sig = np.mean(x ** 2)
    p_noise = p_sig / (10 ** (snr_db / 10.0))
    return x + rng.normal(0, np.sqrt(p_noise), size=x.shape)


def _fit_eval(tr, te, keys, seed=42):
    scaler = StandardScaler().fit(tr[keys])
    clf = RandomForestClassifier(n_estimators=300, random_state=seed)
    clf.fit(scaler.transform(tr[keys]), tr.label)
    return accuracy_score(te.label, clf.predict(scaler.transform(te[keys])))


def noise_sweep(X, y, load, rpm, snr_list=(20, 15, 10, 5, 0, -5, -10),
                test_load=1, seed=42, verbose=True):
    """Train on clean signals, test on noisy ones.

    Deliberately asymmetric: a model is trained in the lab and deployed into a
    noisy plant. Training clean and testing noisy mirrors that, and is harder
    (and more honest) than training on noise too.
    """
    rng = np.random.default_rng(seed)
    sets = {'time only': TIME_KEYS, 'frequency only': FREQ_KEYS,
            'both': TIME_KEYS + FREQ_KEYS}

    clean = pd.DataFrame([{**extract(x, FS, r), 'label': l, 'load': ld}
                          for x, l, ld, r in zip(X, y, load, rpm)])
    tr = clean[clean.load != test_load]

    rows = []
    for snr in snr_list:
        noisy = pd.DataFrame([
            {**extract(add_noise(x, snr, rng), FS, r), 'label': l, 'load': ld}
            for x, l, ld, r in zip(X, y, load, rpm) if ld == test_load])
        row = {'snr_db': snr}
        for name, keys in sets.items():
            row[name] = _fit_eval(tr, noisy, keys, seed)
        rows.append(row)
        if verbose:
            print(f"  SNR {snr:+4d} dB  " +
                  "  ".join(f"{k}={row[k]:.3f}" for k in sets))
    return pd.DataFrame(rows)


def window_sweep(signals, sizes=(2048, 1024, 512, 256, 128),
                 test_load=1, seed=42, verbose=True):
    """Re-window the raw recordings at several lengths and re-run.

    signals: list of (raw_signal, label, load, rpm) tuples.
    """
    from data import window
    sets = {'time only': TIME_KEYS, 'frequency only': FREQ_KEYS,
            'both': TIME_KEYS + FREQ_KEYS}
    rows = []
    for size in sizes:
        recs = []
        for sig, lab, ld, r in signals:
            for w in window(sig, size=size, overlap=0.5):
                recs.append({**extract(w, FS, r), 'label': lab, 'load': ld})
        df = pd.DataFrame(recs)
        tr, te = df[df.load != test_load], df[df.load == test_load]
        row = {'window': size, 'ms': round(1000 * size / FS, 1),
               'n_windows': len(df)}
        for name, keys in sets.items():
            row[name] = _fit_eval(tr, te, keys, seed)
        rows.append(row)
        if verbose:
            print(f"  {size:5d} samples ({row['ms']:6.1f} ms)  " +
                  "  ".join(f"{k}={row[k]:.3f}" for k in sets))
    return pd.DataFrame(rows)

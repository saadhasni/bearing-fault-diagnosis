"""Cross-load generalisation experiment.

THE QUESTION
------------
Which features still work when the motor load changes?

PROTOCOL
--------
Leave-one-load-out. Train on three motor loads, test on the fourth, rotate over
all four, and report mean +/- std. This is far more honest than a single split:
it shows whether performance is stable or depends on which condition was held
out.

WHAT TO EXPECT
--------------
Time-domain statistics (rms, peak, std) scale with vibration amplitude, which
scales with load - so they should degrade when tested on an unseen load.
Envelope features are normalised by total envelope energy and evaluated at
defect frequencies recomputed for each shaft speed, so they should hold up.

If time-domain features still score ~100% here, something is still leaking.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from features import extract, TIME_KEYS, FREQ_KEYS


def build_feature_table(X, y, load, rpm, fs, band=(2000, 6000), verbose=True):
    """Extract features for every window.

    NOTE: rpm is passed PER WINDOW, so defect frequencies are recomputed for
    each operating condition. This is essential - BPFO at 1730 rpm differs from
    BPFO at 1797 rpm by about 4 Hz, which is wider than the integration band.
    """
    rows = []
    n = len(X)
    for i, (xi, yi, li, ri) in enumerate(zip(X, y, load, rpm)):
        f = extract(xi, fs, ri, band)
        f['label'] = yi
        f['load'] = li
        f['rpm'] = ri
        rows.append(f)
        if verbose and (i + 1) % 500 == 0:
            print(f"    {i+1}/{n} windows")
    return pd.DataFrame(rows)


def run_fold(df, feature_keys, test_load, model='rf', seed=42):
    """Train on all loads except test_load; evaluate on test_load."""
    tr = df[df.load != test_load]
    te = df[df.load == test_load]
    if len(tr) == 0 or len(te) == 0:
        raise ValueError(
            f"Cross-load split failed: holding out {test_load}HP leaves "
            f"{len(tr)} training and {len(te)} test windows.\n"
            f"Loads present in the data: {sorted(df.load.unique())}\n"
            "You need at least TWO motor loads. Download the 1, 2 and 3 HP "
            "files listed in data.py FILES.")

    scaler = StandardScaler().fit(tr[feature_keys])
    Xtr, Xte = scaler.transform(tr[feature_keys]), scaler.transform(te[feature_keys])

    clf = (RandomForestClassifier(n_estimators=300, random_state=seed)
           if model == 'rf' else
           SVC(kernel='rbf', C=10, gamma='scale', random_state=seed))
    clf.fit(Xtr, tr.label)
    pred = clf.predict(te[feature_keys].pipe(lambda d: scaler.transform(d)))

    labels = sorted(df.label.unique())
    return {
        'test_load': test_load,
        'accuracy': accuracy_score(te.label, pred),
        'confusion': confusion_matrix(te.label, pred, labels=labels),
        'labels': labels,
        'importances': (dict(zip(feature_keys, clf.feature_importances_))
                        if model == 'rf' else None),
        'y_true': te.label.values,
        'y_pred': pred,
    }


def leave_one_load_out(df, feature_keys, model='rf'):
    """Rotate the held-out load over all four conditions."""
    folds = [run_fold(df, feature_keys, L, model) for L in sorted(df.load.unique())]
    accs = np.array([f['accuracy'] for f in folds])
    total_cm = sum(f['confusion'] for f in folds)
    return {
        'per_fold': {int(f['test_load']): float(f['accuracy']) for f in folds},
        'mean': float(accs.mean()),
        'std': float(accs.std()),
        'confusion': total_cm,
        'labels': folds[0]['labels'],
        'folds': folds,
    }


def compare(df, model='rf'):
    """All three feature sets under the cross-load protocol."""
    sets = {'time only': TIME_KEYS,
            'frequency only': FREQ_KEYS,
            'both': TIME_KEYS + FREQ_KEYS}
    results = {name: leave_one_load_out(df, keys, model)
               for name, keys in sets.items()}
    summary = pd.DataFrame([{
        'feature set': name,
        'n features': len(sets[name]),
        'mean acc': round(r['mean'], 3),
        'std': round(r['std'], 3),
        **{f'{L}HP': round(a, 3) for L, a in r['per_fold'].items()},
    } for name, r in results.items()])
    return summary, results


def within_load_baseline(df, feature_keys, model='rf', seed=42):
    """The FLAWED protocol, kept deliberately for comparison.

    Random split ignoring load. Reproduces the ~100% result and demonstrates in
    the README why it was misleading. Showing the flawed and corrected numbers
    side by side is more persuasive than only reporting the corrected one.
    """
    from sklearn.model_selection import train_test_split
    Xtr, Xte, ytr, yte = train_test_split(
        df[feature_keys], df.label, test_size=0.3,
        random_state=seed, stratify=df.label)
    scaler = StandardScaler().fit(Xtr)
    clf = (RandomForestClassifier(n_estimators=300, random_state=seed)
           if model == 'rf' else SVC(kernel='rbf', C=10, random_state=seed))
    clf.fit(scaler.transform(Xtr), ytr)
    return accuracy_score(yte, clf.predict(scaler.transform(Xte)))

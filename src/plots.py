"""Generate all figures for the README.

Run from the project root AFTER main.py and run_robustness.py:
    python src/plots.py

Writes PNGs into results/.
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from data import FS, FILES, RPM_BY_LOAD, load_de_signal, window
from features import envelope_spectrum, defect_frequencies

RESULTS = Path('results')
DATA = Path('data')

# consistent styling across all figures
COLOURS = {'time only': '#C44E52', 'frequency only': '#4C72B0', 'both': '#55A868'}
plt.rcParams.update({
    'font.size': 10, 'axes.grid': True, 'grid.alpha': 0.3,
    'figure.dpi': 150, 'savefig.bbox': 'tight', 'axes.spines.top': False,
    'axes.spines.right': False,
})


def fig_noise_sweep():
    """Accuracy vs SNR, with the majority-class baseline marked."""
    df = pd.read_csv(RESULTS / 'noise_sweep.csv')
    feat = pd.read_csv(RESULTS / 'features.csv')
    te = feat[feat.load == 1]
    baseline = te.label.value_counts().max() / len(te)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for col in ('time only', 'frequency only', 'both'):
        ax.plot(df['snr_db'], df[col], 'o-', label=col, color=COLOURS[col], lw=2, ms=6)
    ax.axhline(baseline, ls='--', c='grey', lw=1.5)
    ax.text(df['snr_db'].min(), baseline + 0.02,
            f'majority-class baseline ({baseline:.3f})', fontsize=9, color='grey')

    ax.set_xlabel('Test SNR (dB)')
    ax.set_ylabel('Accuracy')
    ax.set_title('Noise robustness: trained on clean signals, tested on noisy',
                 fontsize=11)
    ax.invert_xaxis()
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False)

    # annotate the two failure modes
    ax.annotate('fails silent\n(predicts all "normal")', xy=(-5, 0.572),
                xytext=(-3, 0.40), fontsize=8, color=COLOURS['frequency only'],
                arrowprops=dict(arrowstyle='->', color=COLOURS['frequency only'], lw=1))
    ax.annotate('fails loud\n(confident wrong faults)', xy=(-5, 0.144),
                xytext=(2, 0.20), fontsize=8, color=COLOURS['time only'],
                arrowprops=dict(arrowstyle='->', color=COLOURS['time only'], lw=1))

    fig.savefig(RESULTS / 'fig_noise_sweep.png')
    plt.close(fig)
    print('  fig_noise_sweep.png')


def fig_window_sweep():
    """Accuracy vs window length, with the FFT-resolution limit marked."""
    df = pd.read_csv(RESULTS / 'window_sweep.csv')
    d = defect_frequencies(1797)
    sep = d['BPFI'] - d['BPFO']              # 54.8 Hz
    df['resolution'] = FS / df['window']

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2))

    for col in ('time only', 'frequency only', 'both'):
        a1.plot(df['window'], df[col], 'o-', label=col, color=COLOURS[col], lw=2, ms=6)
    a1.set_xscale('log', base=2)
    a1.set_xticks(df['window'])
    a1.set_xticklabels(df['window'])
    a1.set_xlabel('Window length (samples)')
    a1.set_ylabel('Accuracy')
    a1.set_title('Accuracy vs analysis window', fontsize=11)
    a1.set_ylim(0.6, 1.05)
    a1.legend(frameon=False)
    for x, ms in zip(df['window'], df['ms']):
        a1.annotate(f'{ms:.0f} ms', (x, 0.63), ha='center', fontsize=7, color='grey')

    a2.plot(df['resolution'], df['frequency only'], 'o-',
            color=COLOURS['frequency only'], lw=2, ms=6, label='frequency only')
    a2.axvline(sep, ls='--', c='k', lw=1.5)
    a2.text(sep * 1.06, 0.95, f'BPFI - BPFO = {sep:.1f} Hz',
            fontsize=9, rotation=90, va='top')
    a2.axvspan(sep, df['resolution'].max() * 1.1, color='red', alpha=0.07)
    a2.text(sep * 1.5, 0.72, 'defect frequencies\nunresolvable', fontsize=8,
            color='darkred')
    a2.set_xscale('log')
    a2.set_xlabel('FFT resolution, $f_s/N$ (Hz)')
    a2.set_ylabel('Accuracy')
    a2.set_title('Envelope features fail once resolution exceeds\nthe defect-frequency spacing',
                 fontsize=11)
    a2.set_ylim(0.6, 1.05)

    fig.savefig(RESULTS / 'fig_window_sweep.png')
    plt.close(fig)
    print('  fig_window_sweep.png')


def fig_envelope_spectrum():
    """Envelope spectra for each class with defect frequencies marked.
    This is the figure that shows the method working."""
    fig, axes = plt.subplots(4, 1, figsize=(8, 9), sharex=True)
    order = ['normal', 'inner_race', 'outer_race', 'ball']
    d = defect_frequencies(RPM_BY_LOAD[0])

    for ax, label in zip(axes, order):
        p = DATA / FILES[label][0]
        if not p.exists():
            ax.text(0.5, 0.5, f'missing {p.name}', ha='center', transform=ax.transAxes)
            continue
        sig = load_de_signal(p)
        w = window(sig, 4096, 0.5)[:20]           # average a few windows
        specs = []
        for x in w:
            f, s = envelope_spectrum(x, FS)
            specs.append(s)
        f = f[f < 500]
        spec = np.mean(specs, axis=0)[:len(f)]
        spec = spec / spec.max()

        ax.plot(f, spec, lw=0.9, color='#333333')
        for name, colour in (('BPFO', '#C44E52'), ('BPFI', '#4C72B0')):
            for h in (1, 2, 3):
                fx = d[name] * h
                if fx < 500:
                    ax.axvline(fx, color=colour, ls='--', lw=1,
                               alpha=0.9 if h == 1 else 0.35)
            ax.plot([], [], color=colour, ls='--', label=f'{name} ({d[name]:.0f} Hz)')
        ax.set_ylabel('norm.\namplitude', fontsize=9)
        ax.set_title(label.replace('_', ' '), fontsize=10, loc='left')
        ax.set_ylim(0, 1.05)
        if label == 'normal':
            ax.legend(frameon=False, fontsize=8, ncol=2)

    axes[-1].set_xlabel('Frequency (Hz)')
    fig.suptitle('Envelope spectra, 0 HP (1797 rpm) — dashed lines mark computed defect frequencies',
                 fontsize=11, y=0.995)
    fig.tight_layout()
    fig.savefig(RESULTS / 'fig_envelope_spectrum.png')
    plt.close(fig)
    print('  fig_envelope_spectrum.png')


def fig_raw_signals():
    """Raw vibration for each class - shows the impulsiveness difference."""
    fig, axes = plt.subplots(4, 1, figsize=(8, 6.5), sharex=True, sharey=True)
    order = ['normal', 'ball', 'inner_race', 'outer_race']
    for ax, label in zip(axes, order):
        p = DATA / FILES[label][0]
        if not p.exists():
            continue
        sig = load_de_signal(p)[:4096]
        t = np.arange(len(sig)) / FS * 1000
        ax.plot(t, sig, lw=0.6, color='#333333')
        ax.set_ylabel('accel.\n(g)', fontsize=9)
        ax.set_title(label.replace('_', ' '), fontsize=10, loc='left')
    axes[-1].set_xlabel('Time (ms)')
    fig.suptitle('Raw drive-end vibration, 0 HP', fontsize=11)
    fig.tight_layout()
    fig.savefig(RESULTS / 'fig_raw_signals.png')
    plt.close(fig)
    print('  fig_raw_signals.png')


def fig_feature_separation():
    """Kurtosis vs envelope BPFO/BPFI - shows why each family works."""
    df = pd.read_csv(RESULTS / 'features.csv')
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2))
    cmap = {'normal': '#55A868', 'ball': '#8172B2',
            'inner_race': '#4C72B0', 'outer_race': '#C44E52'}

    for label, g in df.groupby('label'):
        # NOTE: bracket notation is required - g.kurtosis would resolve to
        # pandas' built-in DataFrame.kurtosis() method, not the column.
        a1.scatter(g['kurtosis'], g['crest_factor'], s=8, alpha=0.5,
                   label=label.replace('_', ' '), color=cmap[label])
        a2.scatter(g['env_BPFO'], g['env_BPFI'], s=8, alpha=0.5,
                   label=label.replace('_', ' '), color=cmap[label])

    a1.set_xlabel('Kurtosis'); a1.set_ylabel('Crest factor')
    a1.set_title('Time-domain features', fontsize=11)
    a1.legend(frameon=False, fontsize=8)
    a2.set_xlabel('Envelope energy at BPFO'); a2.set_ylabel('Envelope energy at BPFI')
    a2.set_title('Envelope-spectrum features', fontsize=11)
    a2.legend(frameon=False, fontsize=8)

    fig.savefig(RESULTS / 'fig_feature_separation.png')
    plt.close(fig)
    print('  fig_feature_separation.png')


if __name__ == '__main__':
    RESULTS.mkdir(exist_ok=True)
    print('Generating figures...')
    for fn in (fig_noise_sweep, fig_window_sweep, fig_feature_separation,
               fig_envelope_spectrum, fig_raw_signals):
        try:
            fn()
        except Exception as e:
            print(f'  SKIPPED {fn.__name__}: {e}')
    print('Done.')

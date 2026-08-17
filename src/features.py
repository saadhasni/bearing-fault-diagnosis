"""Feature extraction for bearing fault diagnosis.

Two feature families, extracted separately so the experiment can compare them:

  TIME DOMAIN   - cheap statistics on the raw vibration signal. These detect
                  THAT something is wrong (an impacting fault makes the signal
                  spikier) but are poor at identifying WHICH component failed.

  FREQUENCY     - envelope spectrum energy at the characteristic defect
                  frequencies. A bearing fault produces periodic impacts whose
                  repetition rate depends on which surface is damaged, so this
                  is what actually distinguishes inner race from outer race
                  from ball faults.

That distinction is the whole point of the project. See README.
"""

import numpy as np
from scipy import signal as sps
from scipy.stats import kurtosis, skew


# ---------------------------------------------------------------- time domain

def time_features(x):
    """Nine classical condition-monitoring statistics.

    Kurtosis and crest factor are the important ones: a healthy bearing gives
    a roughly Gaussian signal (kurtosis ~3), while an impacting fault adds
    sharp transients that push kurtosis well above 3.
    """
    x = np.asarray(x, dtype=np.float64)
    rms = np.sqrt(np.mean(x ** 2))
    peak = np.max(np.abs(x))
    mean_abs = np.mean(np.abs(x))
    return {
        'rms': rms,
        'peak': peak,
        'kurtosis': kurtosis(x, fisher=False),      # 3.0 for Gaussian
        'skewness': skew(x),
        'crest_factor': peak / rms if rms > 0 else 0.0,
        'shape_factor': rms / mean_abs if mean_abs > 0 else 0.0,
        'impulse_factor': peak / mean_abs if mean_abs > 0 else 0.0,
        'clearance_factor': peak / (np.mean(np.sqrt(np.abs(x))) ** 2 + 1e-12),
        'std': np.std(x),
    }


# ----------------------------------------------------------- defect frequency

def defect_frequencies(rpm, n_balls=9, ball_dia=0.3126, pitch_dia=1.537,
                       contact_angle=0.0):
    """Characteristic bearing defect frequencies, in Hz.

    Defaults are the SKF 6205-2RS deep groove bearing used at the drive end in
    the CWRU rig. These come from the bearing geometry, not from the data -
    which is exactly why this is an engineering approach rather than a purely
    statistical one.

    BPFO - ball pass frequency, outer race
    BPFI - ball pass frequency, inner race
    BSF  - ball spin frequency
    FTF  - fundamental train (cage) frequency
    """
    fr = rpm / 60.0                       # shaft rotation, Hz
    ratio = (ball_dia / pitch_dia) * np.cos(contact_angle)
    return {
        'shaft': fr,
        'BPFO': (n_balls / 2.0) * fr * (1 - ratio),
        'BPFI': (n_balls / 2.0) * fr * (1 + ratio),
        'BSF':  (pitch_dia / (2 * ball_dia)) * fr * (1 - ratio ** 2),
        'FTF':  0.5 * fr * (1 - ratio),
    }


# ----------------------------------------------------------- envelope spectrum

def envelope_spectrum(x, fs, band=(2000, 6000)):
    """Band-pass, then Hilbert envelope, then FFT.

    Why not just FFT the raw signal? Because bearing impacts EXCITE a
    high-frequency structural resonance rather than appearing directly at the
    defect frequency. The defect frequency shows up as the MODULATION of that
    resonance. Band-passing around the resonance and taking the envelope
    demodulates it, moving the defect frequency into the low-frequency
    spectrum where it is visible.

    This is the single most important signal processing step in the project.
    """
    x = np.asarray(x, dtype=np.float64)
    nyq = fs / 2.0
    lo, hi = band[0] / nyq, min(band[1] / nyq, 0.99)
    b, a = sps.butter(4, [lo, hi], btype='band')
    filtered = sps.filtfilt(b, a, x)
    env = np.abs(sps.hilbert(filtered))
    env = env - env.mean()                       # drop DC
    spec = np.abs(np.fft.rfft(env * np.hanning(len(env))))
    freqs = np.fft.rfftfreq(len(env), d=1.0 / fs)
    return freqs, spec


def band_energy(freqs, spec, centre, width=5.0, harmonics=3):
    """Total spectral energy in narrow bands around a defect frequency and its
    first few harmonics. Harmonics matter - a real fault produces a series, not
    a single line, so summing them is more robust than reading one bin."""
    total = 0.0
    for h in range(1, harmonics + 1):
        f0 = centre * h
        if f0 >= freqs[-1]:
            break
        m = (freqs >= f0 - width) & (freqs <= f0 + width)
        if m.any():
            total += float(np.sum(spec[m] ** 2))
    return total


def freq_features(x, fs, rpm, band=(2000, 6000)):
    """Envelope-spectrum energy at each defect frequency, normalised by total
    envelope energy so the features are amplitude-independent."""
    freqs, spec = envelope_spectrum(x, fs, band)
    total = float(np.sum(spec ** 2)) + 1e-12
    d = defect_frequencies(rpm)
    out = {}
    for name in ('BPFO', 'BPFI', 'BSF', 'FTF', 'shaft'):
        out[f'env_{name}'] = band_energy(freqs, spec, d[name]) / total
    # spectral shape descriptors
    p = spec ** 2 / total
    out['spec_centroid'] = float(np.sum(freqs * p))
    out['spec_entropy'] = float(-np.sum(p * np.log(p + 1e-12)))
    return out


def extract(x, fs, rpm, band=(2000, 6000)):
    """All features for one signal window."""
    f = time_features(x)
    f.update(freq_features(x, fs, rpm, band))
    return f


TIME_KEYS = ['rms', 'peak', 'kurtosis', 'skewness', 'crest_factor',
             'shape_factor', 'impulse_factor', 'clearance_factor', 'std']
FREQ_KEYS = ['env_BPFO', 'env_BPFI', 'env_BSF', 'env_FTF', 'env_shaft',
             'spec_centroid', 'spec_entropy']

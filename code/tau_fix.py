"""
tau_fix.py — corrected diastolic time-constant estimator.

Two changes from the original.

1. WINDOW SELECTION
   The old dicrotic-notch detector fired on an inflection of the systolic
   downstroke (0.215 s in case 7, versus a true notch near 0.35 s). The fit
   window therefore covered the downstroke rather than the diastolic decay.
   The notch is now located as the first local MINIMUM of pressure after the
   point of steepest descent, with a further skip to clear the dicrotic wave.

2. ESTIMATOR
   For P(t) = A e^{-t/tau} + P_inf,

        dP/dt = -(A/tau) e^{-t/tau}
        ln(-dP/dt) = const - t/tau

   so a linear regression of ln(-dP/dt) on t recovers tau with slope -1/tau,
   independent of the asymptote P_inf. Under an affine transduction
   y = kP + b the same regression gives the same slope, since the gain enters
   only the intercept. This removes BOTH nuisance parameters that were
   trading off against tau, and makes pressure and plethysmographic channels
   directly comparable without calibration.

   Differentiation amplifies noise, so the signal is smoothed with a
   Savitzky-Golay filter before the derivative is taken.
"""

import numpy as np
from scipy.signal import savgol_filter, find_peaks
from vitaldb_pipeline import lowpass

FS = 500.0


# --------------------------------------------------------------------------
def diastole_start(seg, fs=FS, skip_dicrotic=0.25):
    """Index at which diastolic decay begins within a foot-to-foot beat.

    Locates the systolic peak, then the point of steepest descent, then the
    first local minimum after it (the dicrotic notch). A further fraction of
    the remaining beat is skipped to clear the dicrotic wave.
    """
    n = len(seg)
    s = lowpass(seg, fs, hi=20.0)
    pk = int(np.argmax(s))
    if n - pk < int(0.15 * fs):
        return None

    d = np.gradient(s)
    tail = d[pk:]
    steep = pk + int(np.argmin(tail))            # most negative slope
    if n - steep < int(0.10 * fs):
        return None

    # first local minimum of pressure after the steepest descent
    post = s[steep:]
    mins, _ = find_peaks(-post, distance=int(0.02 * fs))
    notch = steep + int(mins[0]) if mins.size else steep + int(0.05 * fs)

    start = notch + int(skip_dicrotic * (n - notch))
    if n - start < int(0.12 * fs):               # need >=120 ms of decay
        return None
    return int(start)


# --------------------------------------------------------------------------
def tau_derivative(seg, fs=FS, min_dur=0.15, min_r2=0.90):
    """Estimate tau from the slope of ln(-dP/dt). Asymptote- and gain-free.

    Returns (tau, r2, n_pts) or (nan, nan, 0).
    """
    i0 = diastole_start(seg, fs)
    if i0 is None:
        return np.nan, np.nan, 0
    tail = np.asarray(seg[i0:], dtype=float)
    if tail.size < max(int(min_dur * fs), 25):
        return np.nan, np.nan, 0

    win = int(0.04 * fs) | 1                     # ~40 ms, odd
    win = min(win, (tail.size // 2) * 2 - 1)
    if win < 5:
        return np.nan, np.nan, 0
    sm = savgol_filter(tail, win, 3)
    d = np.gradient(sm, 1.0 / fs)

    m = d < 0                                    # decay only
    if m.sum() < 20:
        return np.nan, np.nan, 0
    t = np.arange(tail.size)[m] / fs
    ln = np.log(-d[m])

    # trim the first and last 5% to avoid edge effects of the filter
    k = max(1, int(0.05 * t.size))
    t, ln = t[k:-k], ln[k:-k]
    if t.size < 15:
        return np.nan, np.nan, 0

    A = np.vstack([t, np.ones_like(t)]).T
    coef, *_ = np.linalg.lstsq(A, ln, rcond=None)
    slope = coef[0]
    if slope >= 0 or not np.isfinite(slope):
        return np.nan, np.nan, 0
    tau = -1.0 / slope
    if not (0.2 < tau < 8.0):
        return np.nan, np.nan, 0

    # Goodness of fit is assessed on the SIGNAL, not on the log-derivative:
    # differentiation amplifies noise, so r2 of ln(-dP/dt) understates fit
    # quality badly even when the slope estimate is sound. With tau fixed,
    # amplitude and asymptote follow by linear least squares.
    ts = np.arange(tail.size) / fs
    B = np.vstack([np.exp(-ts / tau), np.ones_like(ts)]).T
    ab, *_ = np.linalg.lstsq(B, tail, rcond=None)
    if ab[0] <= 0:
        return np.nan, np.nan, 0
    fit = B @ ab
    ss_res = float(np.sum((tail - fit) ** 2))
    ss_tot = float(np.sum((tail - tail.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    if not np.isfinite(r2) or r2 < min_r2:
        return np.nan, np.nan, 0
    return float(tau), float(r2), int(tail.size)


def tau_from_signal_v2(x, fs=FS, min_r2=0.90, max_beats=None):
    """Beat-wise tau using the corrected window and derivative estimator."""
    from vitaldb_pipeline import detect_feet
    feet = detect_feet(x, fs)
    if feet.size < 3:
        return np.array([]), np.array([])
    taus, r2s = [], []
    pairs = list(zip(feet[:-1], feet[1:]))
    if max_beats:
        pairs = pairs[:max_beats]
    for a, b in pairs:
        tau, r2, npts = tau_derivative(x[a:b], fs, min_r2=min_r2)
        if np.isfinite(tau):
            taus.append(tau); r2s.append(r2)
    return np.asarray(taus), np.asarray(r2s)


# ==========================================================================
if __name__ == "__main__":
    from wk_sim import synth_case
    from vitaldb_pipeline import robust_tau

    rng = np.random.default_rng(5)
    print("=" * 70)
    print("Validation of corrected estimator on synthetic ground truth")
    print("=" * 70)
    print(f"{'tau true':>9} {'tau hat':>9} {'err %':>8} {'beats':>7}")
    errs = []
    for tau_t in [1.0, 1.3, 1.6, 1.9, 2.2]:
        R = 1.1
        d = synth_case(R, tau_t / R, level="L0", gain=1.0, offset=0.0,
                       rng=rng, snr_db=40.0, fs=FS, n_beats=16, keep_last=10)
        taus, _ = tau_from_signal_v2(d["P"], FS)
        th, sd, n = robust_tau(taus)
        e = 100 * (th - tau_t) / tau_t
        errs.append(abs(e))
        print(f"{tau_t:9.3f} {th:9.3f} {e:+8.2f} {n:7d}")
    print(f"\n   mean |err| = {np.mean(errs):.2f}%")

    print("\n" + "=" * 70)
    print("Gain invariance (the key property)")
    print("=" * 70)
    print(f"{'gain':>8} {'tau(P)':>9} {'tau(y)':>9} {'diff %':>8}")
    for k in [0.3, 1.0, 3.0, 8.0]:
        d = synth_case(1.1, 1.45, level="L0", gain=k,
                       offset=float(rng.uniform(-5, 5)), rng=rng,
                       snr_db=40.0, fs=FS, n_beats=16, keep_last=10)
        a, _, _ = robust_tau(tau_from_signal_v2(d["P"], FS)[0])
        p, _, _ = robust_tau(tau_from_signal_v2(d["y"], FS)[0])
        print(f"{k:8.2f} {a:9.3f} {p:9.3f} {100*(p-a)/a:+8.2f}")

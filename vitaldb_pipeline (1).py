"""
vitaldb_pipeline.py — waveform pipeline for the PPG identifiability study.

Run in Colab:
    !pip install -q vitaldb
    from vitaldb_pipeline import *

Two experiments, different cohorts:

  EXP-A  (n ~ 3459)   ART + PLETH
         tau from ABP diastolic decay  vs  tau from PPG diastolic decay.
         Prediction: they AGREE. tau is identifiable under L0.
         Also yields the empirical PPG<->ABP transduction map per patient,
         which tests how far real physiology sits from the affine case.

  EXP-B  (subset)     ART + PLETH + cardiac output
         Reference R = MAP/CO, C = tau/R  (calibrated, uses flow).
         PPG-derived R, C from the same recording.
         Prediction: they DISAGREE, and the disagreement lies along the
         tau isoline.

Note: R and C cannot be separated from pressure alone either -- flow is
required. EXP-B is therefore gated on CO availability, which must be
counted before committing.
"""

import numpy as np
from scipy.signal import find_peaks, butter, filtfilt
from scipy.optimize import curve_fit

FS = 500.0


# ==========================================================================
# Cohort discovery
# ==========================================================================

CO_TRACKS = ["EV1000/CO", "EV1000/CI", "Vigileo/CO", "Vigileo/CI",
             "CardioQ/CO", "CardioQ/CI", "Solar8000/CO"]


def find_cohorts(trks_url="https://api.vitaldb.net/trks"):
    """Return case-id sets for EXP-A and EXP-B."""
    import pandas as pd
    df = pd.read_csv(trks_url)
    art = set(df[df.tname == "SNUADC/ART"].caseid)
    pleth = set(df[df.tname == "SNUADC/PLETH"].caseid)
    exp_a = art & pleth

    co_avail = {}
    for tr in CO_TRACKS:
        ids = set(df[df.tname == tr].caseid)
        if ids:
            co_avail[tr] = ids & exp_a
    exp_b = set().union(*co_avail.values()) if co_avail else set()

    print(f"EXP-A (ART + PLETH)      : {len(exp_a)}")
    for tr, ids in sorted(co_avail.items(), key=lambda x: -len(x[1])):
        print(f"   + {tr:18s} : {len(ids)}")
    print(f"EXP-B (any CO track)     : {len(exp_b)}")
    return exp_a, exp_b, co_avail


def load_waveforms(caseid, tracks=("SNUADC/ART", "SNUADC/PLETH"), fs=FS):
    import vitaldb
    v = vitaldb.load_case(caseid, list(tracks), 1.0 / fs)
    return v


# ==========================================================================
# Preprocessing and quality control
# ==========================================================================

def bandpass(x, fs=FS, lo=0.5, hi=10.0, order=2):
    b, a = butter(order, [lo / (fs / 2), hi / (fs / 2)], btype="band")
    return filtfilt(b, a, x)


def lowpass(x, fs=FS, hi=12.0, order=4):
    b, a = butter(order, hi / (fs / 2), btype="low")
    return filtfilt(b, a, x)


def segment_quality(abp, ppg, fs=FS,
                    abp_range=(30.0, 220.0), max_nan_frac=0.02,
                    min_pp=15.0, max_flat_s=0.5):
    """Boolean pass/fail plus a reason string for one analysis window."""
    if abp is None or ppg is None:
        return False, "missing channel"
    if np.mean(np.isnan(abp)) > max_nan_frac or np.mean(np.isnan(ppg)) > max_nan_frac:
        return False, "nan fraction"
    a = abp[~np.isnan(abp)]
    p = ppg[~np.isnan(ppg)]
    if a.size < fs * 10 or p.size < fs * 10:
        return False, "too short"
    if a.min() < abp_range[0] or a.max() > abp_range[1]:
        return False, "abp out of range"
    if np.ptp(a) < min_pp:
        return False, "pulse pressure too small"
    # flatline / clipping detection
    for sig, name in ((a, "abp"), (p, "ppg")):
        d = np.abs(np.diff(sig))
        flat = d < 1e-9
        if flat.size:
            runs = np.diff(np.flatnonzero(np.diff(np.r_[0, flat, 0])))[::2]
            if runs.size and runs.max() > max_flat_s * fs:
                return False, f"{name} flatline"
    return True, "ok"


# ==========================================================================
# Beat segmentation
# ==========================================================================

def detect_feet(x, fs=FS, min_hr=35.0, max_hr=180.0):
    """Diastolic feet as local minima with a physiological refractory period."""
    xf = lowpass(x, fs)
    min_dist = int(fs * 60.0 / max_hr)
    idx, _ = find_peaks(-xf, distance=min_dist,
                        prominence=0.15 * np.ptp(xf))
    if idx.size < 3:
        return np.array([], dtype=int)
    # drop implausible intervals
    rr = np.diff(idx) / fs
    ok = (rr > 60.0 / max_hr) & (rr < 60.0 / min_hr)
    keep = np.r_[True, ok]
    return idx[keep]


def dicrotic_notch(seg, fs=FS):
    """Index of the dicrotic notch within one beat (foot-to-foot segment).

    Falls back to a fixed fraction of the beat when no notch is detectable.
    """
    pk = int(np.argmax(seg))
    tail = seg[pk:]
    if tail.size < int(0.06 * fs):
        return min(pk + int(0.25 * len(seg)), len(seg) - 2)
    d2 = np.gradient(np.gradient(lowpass(tail, fs)))
    cand, _ = find_peaks(d2, distance=int(0.03 * fs))
    cand = cand[(cand > int(0.03 * fs)) & (cand < int(0.6 * len(tail)))]
    if cand.size:
        return pk + int(cand[0])
    return min(pk + int(0.25 * len(seg)), len(seg) - 2)


# ==========================================================================
# Diastolic decay fitting  ->  tau
# ==========================================================================

def _exp3(t, A, tau, Pinf):
    return A * np.exp(-t / tau) + Pinf


def fit_tau_beat(seg, fs=FS, skip_frac=0.10, min_pts=25,
                 tau_bounds=(0.20, 20.0), fit_asymptote=True):
    """Fit an exponential to the diastolic tail of one beat.

    Returns (tau, r2, n_pts) or (nan, nan, 0) on failure.
    Works on either ABP (mmHg) or PPG (a.u.) -- under an affine transduction
    the recovered tau is identical, which is the hypothesis under test.
    """
    n = len(seg)
    i0 = dicrotic_notch(seg, fs)
    i0 = i0 + int(skip_frac * (n - i0))          # skip the notch transient
    tail = seg[i0:]
    if tail.size < min_pts:
        return np.nan, np.nan, 0
    t = np.arange(tail.size) / fs
    A0 = max(tail[0] - tail[-1], 1e-6)
    try:
        if fit_asymptote:
            p0 = [A0, 1.5, tail[-1]]
            lo = [1e-9, tau_bounds[0], -np.inf]
            hi = [np.inf, tau_bounds[1], np.inf]
        else:
            p0 = [tail[0], 1.5, 0.0]
            lo = [1e-9, tau_bounds[0], -1e-9]
            hi = [np.inf, tau_bounds[1], 1e-9]
        popt, _ = curve_fit(_exp3, t, tail, p0=p0, bounds=(lo, hi),
                            maxfev=6000)
    except Exception:
        return np.nan, np.nan, 0
    pred = _exp3(t, *popt)
    ss_res = np.sum((tail - pred) ** 2)
    ss_tot = np.sum((tail - tail.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    tau = float(popt[1])
    # flag boundary solutions: a fit that rails is not an estimate
    rel = 1e-3 * (tau_bounds[1] - tau_bounds[0])
    if (tau - tau_bounds[0] < rel) or (tau_bounds[1] - tau < rel):
        return np.nan, np.nan, 0
    return tau, float(r2), int(tail.size)


def tau_from_signal(x, fs=FS, min_r2=0.95, max_beats=None):
    """Beat-wise tau estimates for a whole segment; returns array of taus."""
    feet = detect_feet(x, fs)
    if feet.size < 3:
        return np.array([]), np.array([])
    taus, r2s = [], []
    pairs = list(zip(feet[:-1], feet[1:]))
    if max_beats:
        pairs = pairs[:max_beats]
    for a, b in pairs:
        tau, r2, npts = fit_tau_beat(x[a:b], fs)
        if np.isfinite(tau) and np.isfinite(r2) and r2 >= min_r2:
            taus.append(tau)
            r2s.append(r2)
    return np.asarray(taus), np.asarray(r2s)


def robust_tau(taus, trim=0.2):
    """Trimmed median of beat-wise estimates."""
    if taus.size == 0:
        return np.nan, np.nan, 0
    lo, hi = np.quantile(taus, [trim / 2, 1 - trim / 2])
    core = taus[(taus >= lo) & (taus <= hi)]
    if core.size == 0:
        core = taus
    return float(np.median(core)), float(np.std(core)), int(core.size)


# ==========================================================================
# Empirical transduction map  (the L0 / L1 / L2 test on real data)
# ==========================================================================

def transduction_map(abp, ppg, fs=FS, n_bins=40):
    """Empirical PPG-vs-ABP relation, aggregated over beats.

    Returns bin centres in mmHg, median PPG per bin, and the R^2 of a linear
    fit. Departure from linearity is the degree to which real physiology sits
    above rung L0 of the hierarchy.
    """
    m = np.isfinite(abp) & np.isfinite(ppg)
    a, p = abp[m], ppg[m]
    if a.size < fs * 5:
        return None
    edges = np.quantile(a, np.linspace(0.02, 0.98, n_bins + 1))
    edges = np.unique(edges)
    if edges.size < 5:
        return None
    ctr, med = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (a >= lo) & (a < hi)
        if sel.sum() > 20:
            ctr.append(0.5 * (lo + hi))
            med.append(np.median(p[sel]))
    ctr, med = np.asarray(ctr), np.asarray(med)
    if ctr.size < 5:
        return None
    coef = np.polyfit(ctr, med, 1)
    pred = np.polyval(coef, ctr)
    ss_res = np.sum((med - pred) ** 2)
    ss_tot = np.sum((med - med.mean()) ** 2)
    r2_lin = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    # curvature: quadratic term relative to linear over the observed range
    q = np.polyfit(ctr, med, 2)
    curv = abs(q[0]) * (ctr.max() - ctr.min()) ** 2 / max(abs(np.ptp(med)), 1e-9)
    return dict(p_bin=ctr, ppg_bin=med, r2_linear=float(r2_lin),
                slope=float(coef[0]), curvature=float(curv))


# ==========================================================================
# Per-case driver
# ==========================================================================

def process_case(caseid, win_s=120.0, fs=FS, skip_start_s=600.0):
    """One analysis window from the middle of the case."""
    v = load_waveforms(caseid, fs=fs)
    if v is None or v.shape[0] < fs * (skip_start_s + win_s):
        return None
    abp_all, ppg_all = v[:, 0], v[:, 1]
    mid = abp_all.shape[0] // 2
    half = int(fs * win_s / 2)
    abp = abp_all[mid - half: mid + half]
    ppg = ppg_all[mid - half: mid + half]

    ok, reason = segment_quality(abp, ppg, fs)
    if not ok:
        return dict(caseid=caseid, ok=False, reason=reason)

    abp = np.nan_to_num(abp, nan=np.nanmedian(abp))
    ppg = np.nan_to_num(ppg, nan=np.nanmedian(ppg))

    t_abp, r2_abp = tau_from_signal(abp, fs)
    t_ppg, r2_ppg = tau_from_signal(ppg, fs)
    tau_a, sd_a, n_a = robust_tau(t_abp)
    tau_p, sd_p, n_p = robust_tau(t_ppg)
    tm = transduction_map(abp, ppg, fs)

    return dict(caseid=caseid, ok=True, reason="ok",
                tau_abp=tau_a, tau_abp_sd=sd_a, n_beats_abp=n_a,
                tau_ppg=tau_p, tau_ppg_sd=sd_p, n_beats_ppg=n_p,
                sbp=float(np.percentile(abp, 99)),
                dbp=float(np.percentile(abp, 1)),
                map_=float(np.mean(abp)),
                r2_linear=None if tm is None else tm["r2_linear"],
                curvature=None if tm is None else tm["curvature"])

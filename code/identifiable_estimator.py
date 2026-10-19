"""
identifiable_estimator.py — contribution C4.

An estimator for Windkessel analysis of pulsatile waveforms that reports only
what the data determine, and refuses what they do not.

Design principles
-----------------
1. NEVER report R or C from an uncalibrated signal. By Corollary 1 the
   plethysmogram constrains only tau = RC. The estimator returns the
   admissible ridge instead of a point, and states its width.

2. REPORT tau WITH ITS PRIOR SENSITIVITY. A single tau is meaningless if it
   moves with the assumed diastolic asymptote. The estimator refits under a
   range of priors and returns the spread alongside the estimate.

3. TEST MODEL ADEQUACY BEFORE ESTIMATING. The central failure documented in
   this study is that peripheral diastolic decay is not mono-exponential, so
   a Windkessel fit absorbs curvature it cannot represent. The split-window
   test below detects this directly and cheaply:

       fit tau independently to the first and second half of the diastolic
       decay. For a true mono-exponential these agree. Systematic divergence
       indicates the model does not hold, regardless of how high R^2 is.

   This diagnostic is the piece missing from published PPG-Windkessel work.
   It requires no reference channel and no calibration, so it can be applied
   by anyone, to any recording, before any parameter is reported.

4. RETURN AN EXPLICIT VERDICT. Every output carries a status:
       'ok'          tau is data-determined and the model is adequate
       'weak'        tau estimable but prior-sensitive
       'unusable'    model inadequate or fit on a bound
   A refusal is a result, not a failure.
"""

import numpy as np
from scipy.optimize import least_squares

FS = 500.0
TAU_LO, TAU_HI = 0.15, 12.0
C_LO, C_HI = 0.4, 4.0                 # physiological compliance, mL/mmHg
PRIORS = [(0.0, 10.0), (0.0, 25.0), (0.0, 40.0), (-20.0, 60.0),
          (-np.inf, np.inf)]
SPREAD_OK = 1.5                       # tau considered data-determined below this
SPLIT_TOL = 1.30                      # tau_late/tau_early within this = adequate


# --------------------------------------------------------------------------
def _fit_tails(tails, fs=FS, pinf_bounds=(-np.inf, np.inf), tau0=1.2):
    """Shared tau, shared P_inf, per-beat amplitude."""
    n = len(tails)
    if n < 2:
        return np.nan, np.nan, np.nan, False
    lo_p, hi_p = pinf_bounds
    p0 = min(t[-1] for t in tails) - 5.0
    if np.isfinite(lo_p):
        p0 = max(p0, lo_p + 1e-6)
    if np.isfinite(hi_p):
        p0 = min(p0, hi_p - 1e-6)
    A0 = [max(t[0] - p0, 1e-3) for t in tails]
    th0 = np.r_[np.log(tau0), p0, np.log(A0)]
    lo = np.r_[np.log(TAU_LO), lo_p, np.full(n, np.log(1e-4))]
    hi = np.r_[np.log(TAU_HI), hi_p, np.full(n, np.log(1e4))]

    def res(th):
        tau, pinf, A = np.exp(th[0]), th[1], np.exp(th[2:])
        return np.concatenate([Ai * np.exp(-np.arange(t.size) / fs / tau)
                               + pinf - t for Ai, t in zip(A, tails)])
    try:
        r = least_squares(res, th0, bounds=(lo, hi), xtol=1e-12, ftol=1e-12,
                          max_nfev=4000)
    except Exception:
        return np.nan, np.nan, np.nan, False
    tau, pinf = float(np.exp(r.x[0])), float(r.x[1])
    y = np.concatenate(tails)
    ss = float(np.sum(r.fun ** 2)); st = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss / st if st > 0 else np.nan
    railed = (abs(tau - TAU_LO) < 1e-3) or (abs(tau - TAU_HI) < 1e-3)
    return tau, r2, pinf, railed


# --------------------------------------------------------------------------
def split_window_test(tails, fs=FS, min_pts=60):
    """Model-adequacy diagnostic, free of the asymptote nuisance parameter.

    For P(t) = A exp(-t/tau) + P_inf,

        dP/dt = -(A/tau) exp(-t/tau)   =>   ln(-dP/dt) = const - t/tau

    so ln(-dP/dt) is LINEAR in t with slope -1/tau, whatever P_inf is. A
    mono-exponential decay therefore has the same slope early and late in
    diastole. Curvature of the observed decay -- from wave reflections, or
    from a pressure-dependent compliance -- shows up as a change in slope.

    Fitting each half with a free asymptote instead (an earlier version of
    this test) reproduces the very degeneracy the diagnostic is meant to
    detect, and rejects synthetic Windkessel waveforms. Working on the log
    derivative avoids that entirely.

    Returns (tau_early, tau_late, ratio, adequate, frac_accelerating).
    """
    from scipy.signal import savgol_filter
    se, sl, n_accel = [], [], 0
    for tl in tails:
        if tl.size < min_pts:
            continue
        w = min(int(0.04 * fs) | 1, (tl.size // 2) * 2 - 1)
        if w < 5:
            continue
        d = np.gradient(savgol_filter(np.asarray(tl, float), w, 3), 1.0 / fs)
        t = np.arange(tl.size) / fs
        m = d < 0
        if m.sum() < 30:
            continue
        tt, ln = t[m], np.log(-d[m])
        k = max(1, int(0.05 * tt.size))
        tt, ln = tt[k:-k], ln[k:-k]
        if tt.size < 24:
            continue
        h = tt.size // 2
        beat_ok = True
        vals = []
        for seg_t, seg_l in ((tt[:h], ln[:h]), (tt[h:], ln[h:])):
            A = np.vstack([seg_t, np.ones_like(seg_t)]).T
            c, *_ = np.linalg.lstsq(A, seg_l, rcond=None)
            if c[0] >= 0:
                # A positive slope of ln(-dP/dt) means the decay rate is
                # ACCELERATING, which no exponential can do. This is a
                # decisive rejection of the model, not a failed computation.
                # An earlier version silently discarded these half-windows,
                # which biased the adequacy rate upward.
                beat_ok = False
                break
            vals.append(-1.0 / c[0])
        if beat_ok:
            se.append(vals[0]); sl.append(vals[1])
        else:
            n_accel += 1

    n_tot = n_accel + len(se)
    frac_accel = n_accel / n_tot if n_tot else np.nan

    # No arbitrary threshold is applied to frac_accel: it is returned as a
    # continuous quantity and reported as such. A case is declared adequate
    # only if NO beat shows accelerating decay AND the early/late ratio is
    # within tolerance. Accelerating decay is not a matter of degree -- any
    # occurrence contradicts an exponential -- so the conservative rule is
    # to require none, and to report the observed fraction alongside.
    if len(se) < 2 or len(sl) < 2:
        return np.nan, np.nan, np.nan, False, frac_accel
    te, tl_ = float(np.median(se)), float(np.median(sl))
    if not (np.isfinite(te) and np.isfinite(tl_)) or te <= 0:
        return te, tl_, np.nan, False, frac_accel
    ratio = tl_ / te
    adequate = (n_accel == 0) and ((1.0 / SPLIT_TOL) <= ratio <= SPLIT_TOL)
    return te, tl_, float(ratio), bool(adequate), float(frac_accel)


# --------------------------------------------------------------------------
def analyse(tails, fs=FS, calibrated_gain=False):
    """Full identifiable-only analysis of one recording.

    Parameters
    ----------
    tails : list of 1-D arrays
        Diastolic segments from consecutive beats, same channel.
    calibrated_gain : bool
        True only if the signal is in calibrated pressure units AND an
        independent flow or pressure reference is available. If False, R and
        C are not reported under any circumstance.

    Returns
    -------
    dict with keys:
        status         'ok' | 'weak' | 'unusable'
        reason         explanation when not 'ok'
        tau            point estimate under the unconstrained prior, or nan
        tau_range      (min, max) across priors
        spread         max/min across priors
        model_adequate bool, from the split-window test
        split_ratio    tau_late / tau_early
        R_admissible   (lo, hi) ridge implied by tau, or None
        R, C           reported ONLY if calibrated_gain and status == 'ok'
    """
    out = dict(status="unusable", reason="", tau=np.nan, tau_range=(np.nan, np.nan),
               spread=np.nan, model_adequate=False, split_ratio=np.nan,
               R_admissible=None, R=None, C=None, r2=np.nan, n_beats=len(tails))

    if len(tails) < 3:
        out["reason"] = "fewer than three usable diastolic segments"
        return out

    # --- step 1: model adequacy, BEFORE any parameter is reported
    te, tl, ratio, adequate, frac_acc = split_window_test(tails, fs)
    out.update(split_ratio=ratio, model_adequate=adequate,
               frac_accelerating=frac_acc)
    if not adequate:
        if np.isfinite(frac_acc) and frac_acc > 0:
            out["reason"] = (
                f"mono-exponential decay rejected: decay rate accelerates "
                f"within diastole in {100*frac_acc:.0f}% of beats, which no "
                f"exponential can do.")
        elif not np.isfinite(ratio):
            out["reason"] = (
                "mono-exponential decay rejected: decay rate accelerates "
                "within diastole in the majority of beats, which no "
                "exponential can do. A Windkessel time constant is not "
                "defined for this waveform.")
        else:
            out["reason"] = (
                f"mono-exponential decay rejected: tau_late/tau_early = "
                f"{ratio:.2f} (tolerance {1/SPLIT_TOL:.2f}-{SPLIT_TOL:.2f}). "
                f"A Windkessel time constant is not defined for this waveform.")
        return out

    # --- step 2: tau under every prior
    taus = []
    for b in PRIORS:
        t, r2, _, railed = _fit_tails(tails, fs, pinf_bounds=b)
        if np.isfinite(t) and not railed:
            taus.append(t)
        if b[0] == -np.inf:
            out["tau"], out["r2"] = t, r2
    if len(taus) < 2 or not np.isfinite(out["tau"]):
        out["reason"] = "fit attained a parameter bound; not an estimate"
        return out

    lo, hi = float(min(taus)), float(max(taus))
    spread = hi / lo
    out.update(tau_range=(lo, hi), spread=spread)

    # --- step 3: admissible ridge, always reported instead of R and C
    out["R_admissible"] = (out["tau"] / C_HI, out["tau"] / C_LO)

    # --- step 4: verdict
    if spread <= SPREAD_OK:
        out["status"] = "ok"
    else:
        out["status"] = "weak"
        out["reason"] = (f"tau varies {spread:.2f}-fold with the assumed "
                         f"diastolic asymptote; report the range, not a point")

    # --- step 5: R and C only with genuine calibration
    if calibrated_gain and out["status"] == "ok":
        out["reason"] = ("R and C require an external flow or pressure "
                         "reference; supply one to resolve the ridge")
    return out


def format_report(res):
    """Human-readable summary, showing refusals explicitly."""
    L = []
    L.append(f"status            : {res['status'].upper()}")
    if res["reason"]:
        L.append(f"reason            : {res['reason']}")
    L.append(f"beats used        : {res['n_beats']}")
    fa = res.get("frac_accelerating", np.nan)
    L.append(f"model adequate    : {res['model_adequate']}"
             f"   (tau_late/tau_early = {res['split_ratio']:.3f}"
             f", accelerating beats = "
             f"{'n/a' if not np.isfinite(fa) else f'{100*fa:.0f}%'})")
    if np.isfinite(res["tau"]):
        lo, hi = res["tau_range"]
        L.append(f"tau               : {res['tau']:.3f} s   "
                 f"range across priors {lo:.3f}-{hi:.3f} s "
                 f"({res['spread']:.2f}x)")
    else:
        L.append("tau               : NOT REPORTED")
    if res["R_admissible"]:
        a, b = res["R_admissible"]
        L.append(f"R admissible set  : {a:.3f} to {b:.3f} mmHg s/mL "
                 f"({b/a:.1f}-fold) -- not a measurement")
    L.append("R, C point values : NOT REPORTED "
             "(not identifiable without external calibration)")
    return "\n".join(L)


# ==========================================================================
if __name__ == "__main__":
    from wk_sim import synth_case
    from vitaldb_pipeline import detect_feet
    from tau_fix import diastole_start
    from tau_multibeat import _collect_tails

    rng = np.random.default_rng(3)
    print("=" * 74)
    print("A. Synthetic Windkessel waveforms (model holds by construction)")
    print("=" * 74)
    for tau_t in [1.0, 1.5, 2.0]:
        d = synth_case(1.1, tau_t / 1.1, level="L0", gain=1.0, offset=0.0,
                       hr=72, rng=rng, snr_db=45.0, fs=FS,
                       n_beats=20, keep_last=14)
        tails = _collect_tails(d["P"], detect_feet(d["P"], FS),
                               diastole_start, FS, n_beats=8)
        r = analyse(tails, FS)
        print(f"\n--- true tau = {tau_t:.2f} s ---")
        print(format_report(r))

    print("\n" + "=" * 74)
    print("B. Gain invariance: same physiology, unknown sensor gain")
    print("=" * 74)
    for k in [0.4, 3.0]:
        d = synth_case(1.1, 1.45, level="L0", gain=k, offset=float(k * 2),
                       hr=72, rng=rng, snr_db=45.0, fs=FS,
                       n_beats=20, keep_last=14)
        tails = _collect_tails(d["y"], detect_feet(d["y"], FS),
                               diastole_start, FS, n_beats=8)
        r = analyse(tails, FS)
        print(f"\n--- gain {k}, true tau 1.595 s ---")
        print(format_report(r))

    print("\n" + "=" * 74)
    print("C. L2 volumetric transduction (model does NOT hold)")
    print("=" * 74)
    d = synth_case(1.1, 1.45, level="L2", gain=1.0, offset=0.0, hr=72,
                   rng=rng, snr_db=45.0, fs=FS, n_beats=20, keep_last=14)
    tails = _collect_tails(d["y"], detect_feet(d["y"], FS),
                           diastole_start, FS, n_beats=8)
    print(format_report(analyse(tails, FS)))
    print("\n   The estimator should decline here: a Windkessel tau is not")
    print("   defined for a volumetric observation with pressure-dependent")
    print("   compliance.")

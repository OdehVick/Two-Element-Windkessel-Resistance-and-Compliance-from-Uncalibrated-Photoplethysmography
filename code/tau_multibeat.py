"""
tau_multibeat.py — joint multi-beat estimation of the diastolic time constant,
plus a bound-sensitivity diagnostic.

Problem
-------
Over a single diastolic window of ~0.3 s and a true tau of ~1.5 s, only ~0.2
time constants are observed. An exponential is then nearly linear, and tau
trades off against the asymptote P_inf almost perfectly. Free three-parameter
fits rail (tau -> 0.15 s with P_inf -> 65 mmHg); constrained fits rail on the
bound, so the reported tau is set by the prior rather than by the data.

Fix
---
Fit N consecutive beats jointly with a SHARED tau and a SHARED P_inf but a
per-beat amplitude A_i:

    P_i(t) = A_i exp(-t/tau) + P_inf,    i = 1..N

Beats differ in duration (RR variability) and amplitude (respiratory
modulation), so the decay is observed over a family of windows with different
start pressures. This supplies the curvature information a single beat lacks,
without adding a parameter that can absorb tau.

Parameter count: N + 2 for ~N x 150 samples.

Diagnostic
----------
bound_sensitivity() refits under several P_inf ranges. If tau moves with the
bound, the estimate is prior-determined and must be reported as such; if it is
stable, the constraint is benign.
"""

import numpy as np
from scipy.optimize import least_squares

FS = 500.0


# --------------------------------------------------------------------------
def _collect_tails(sig, feet, start_fn, fs=FS, n_beats=8, min_pts=40):
    """Diastolic tails from consecutive beats."""
    tails = []
    for a, b in zip(feet[:-1], feet[1:]):
        if len(tails) >= n_beats:
            break
        beat = np.asarray(sig[a:b], dtype=float)
        i0 = start_fn(beat, fs)
        if i0 is None:
            continue
        tail = beat[i0:]
        if tail.size >= min_pts:
            tails.append(tail)
    return tails


def tau_joint(tails, fs=FS, tau0=1.2, pinf_bounds=(-np.inf, np.inf)):
    """Joint fit: shared tau and P_inf, per-beat amplitude.

    Returns (tau, r2, n_beats, pinf, hit_bound).
    """
    n = len(tails)
    if n < 2:
        return np.nan, np.nan, 0, np.nan, False

    lo_p, hi_p = pinf_bounds
    p_init = min(max(min(t[-1] for t in tails) - 5.0, lo_p + 1e-6), hi_p - 1e-6) \
        if np.isfinite(lo_p) and np.isfinite(hi_p) else \
        min(t[-1] for t in tails) - 5.0

    # theta = [ln tau, P_inf, ln A_1 .. ln A_n]
    A0 = [max(t[0] - p_init, 1e-3) for t in tails]
    theta0 = np.r_[np.log(tau0), p_init, np.log(A0)]

    lo = np.r_[np.log(0.15), lo_p, np.full(n, np.log(1e-4))]
    hi = np.r_[np.log(12.0), hi_p, np.full(n, np.log(1e4))]

    def resid(th):
        tau = np.exp(th[0]); pinf = th[1]; A = np.exp(th[2:])
        out = []
        for Ai, tl in zip(A, tails):
            t = np.arange(tl.size) / fs
            out.append(Ai * np.exp(-t / tau) + pinf - tl)
        return np.concatenate(out)

    try:
        r = least_squares(resid, theta0, bounds=(lo, hi),
                          xtol=1e-12, ftol=1e-12, max_nfev=4000)
    except Exception:
        return np.nan, np.nan, 0, np.nan, False

    tau = float(np.exp(r.x[0]))
    pinf = float(r.x[1])
    y = np.concatenate(tails)
    ss_res = float(np.sum(r.fun ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    tol = 1e-3 * max(1.0, abs(hi_p - lo_p) if np.isfinite(hi_p - lo_p) else 1.0)
    hit = (np.isfinite(lo_p) and abs(pinf - lo_p) < tol) or \
          (np.isfinite(hi_p) and abs(pinf - hi_p) < tol)
    return tau, float(r2), n, pinf, bool(hit)


def bound_sensitivity(tails, fs=FS,
                      ranges=((0, 10), (0, 25), (0, 40), (-20, 60),
                              (-np.inf, np.inf))):
    """Refit under several P_inf priors. Spread in tau measures how much the
    estimate is determined by the prior rather than the data."""
    out = []
    for lo, hi in ranges:
        tau, r2, n, pinf, hit = tau_joint(tails, fs, pinf_bounds=(lo, hi))
        out.append(dict(lo=lo, hi=hi, tau=tau, r2=r2, pinf=pinf, hit=hit))
    taus = [o["tau"] for o in out if np.isfinite(o["tau"])]
    spread = (max(taus) / min(taus)) if len(taus) > 1 and min(taus) > 0 else np.nan
    return out, spread


# ==========================================================================
if __name__ == "__main__":
    from wk_sim import synth_case
    from vitaldb_pipeline import detect_feet
    from tau_fix import diastole_start
    from scipy.optimize import curve_fit

    rng = np.random.default_rng(9)
    _f = lambda t, A, tau, P: A * np.exp(-t / tau) + P

    def tau_single(tail, fs, bounds):
        t = np.arange(tail.size) / fs
        try:
            p, _ = curve_fit(_f, t, tail,
                             p0=[tail[0] - tail[-1], 1.2, bounds[0]],
                             bounds=([1e-9, 0.15, bounds[0]],
                                     [np.inf, 12.0, bounds[1]]), maxfev=9000)
            return float(p[1]), float(p[2])
        except Exception:
            return np.nan, np.nan

    print("=" * 78)
    print("Single-beat vs joint multi-beat, realistic diastolic window")
    print("=" * 78)
    print(f"{'tau true':>9} {'HR':>5} {'diast(s)':>9} | "
          f"{'single free':>12} {'single(0,25)':>13} | "
          f"{'joint free':>11} {'joint(0,25)':>12}")

    for tau_t, hr in [(1.0, 75), (1.5, 75), (1.5, 60), (2.0, 75), (2.0, 60)]:
        R = 1.1
        d = synth_case(R, tau_t / R, level="L0", gain=1.0, offset=0.0,
                       hr=hr, rng=rng, snr_db=45.0, fs=FS,
                       n_beats=20, keep_last=14)
        feet = detect_feet(d["P"], FS)
        tails = _collect_tails(d["P"], feet, diastole_start, FS, n_beats=8)
        if len(tails) < 3:
            print(f"{tau_t:9.2f} {hr:5d}  insufficient beats")
            continue
        dur = np.mean([t.size for t in tails]) / FS

        sf = np.median([tau_single(t, FS, (-50, 120))[0] for t in tails])
        sc = np.median([tau_single(t, FS, (0, 25))[0] for t in tails])
        jf, _, _, pf, _ = tau_joint(tails, FS)
        jc, _, _, pc, hitc = tau_joint(tails, FS, pinf_bounds=(0, 25))

        print(f"{tau_t:9.2f} {hr:5d} {dur:9.3f} | "
              f"{sf:12.3f} {sc:13.3f} | {jf:11.3f} {jc:12.3f}")

    print("\n" + "=" * 78)
    print("Bound sensitivity, joint fit (true tau = 1.50 s, HR 75)")
    print("=" * 78)
    d = synth_case(1.1, 1.5 / 1.1, level="L0", gain=1.0, offset=0.0,
                   hr=75, rng=rng, snr_db=45.0, fs=FS,
                   n_beats=20, keep_last=14)
    feet = detect_feet(d["P"], FS)
    tails = _collect_tails(d["P"], feet, diastole_start, FS, n_beats=8)
    res, spread = bound_sensitivity(tails, FS)
    print(f"{'P_inf range':>18} {'tau':>8} {'P_inf':>8} {'r2':>8} {'on bound':>9}")
    for o in res:
        rng_s = f"({o['lo']}, {o['hi']})"
        print(f"{rng_s:>18} {o['tau']:8.3f} {o['pinf']:8.2f} "
              f"{o['r2']:8.4f} {str(o['hit']):>9}")
    print(f"\n   tau spread across priors: {spread:.2f}x")

    print("\n" + "=" * 78)
    print("Gain invariance of the joint estimator")
    print("=" * 78)
    print(f"{'gain':>7} {'offset':>8} {'tau(P)':>9} {'tau(y)':>9} {'diff %':>8}")
    for k, off in [(0.3, -3.0), (1.0, 0.0), (3.0, 5.0), (8.0, -10.0)]:
        d = synth_case(1.1, 1.45, level="L0", gain=k, offset=off, hr=75,
                       rng=rng, snr_db=45.0, fs=FS, n_beats=20, keep_last=14)
        fP = detect_feet(d["P"], FS); fy = detect_feet(d["y"], FS)
        tP = _collect_tails(d["P"], fP, diastole_start, FS, 8)
        ty = _collect_tails(d["y"], fy, diastole_start, FS, 8)
        if len(tP) < 3 or len(ty) < 3:
            continue
        a, *_ = tau_joint(tP, FS)
        b, *_ = tau_joint(ty, FS)
        print(f"{k:7.2f} {off:8.1f} {a:9.3f} {b:9.3f} {100*(b-a)/a:+8.2f}")

"""
segmentation_sensitivity.py — reviewer response R1.4 (reproducibility of the
diastolic-onset rule) and R3.3 (is the 99.2% decline an artefact of noise,
filtering or segment selection?).

Each recording is loaded once and analysed under every configuration below,
changing one setting at a time from the published pipeline:

    base            published settings
    skip0.15/0.35   fraction of the post-notch beat skipped (published 0.25)
    shift-20/+20ms  diastolic onset moved 20 ms earlier / later
    onsetLP15/25    low-pass cutoff used to locate the onset (published 20 Hz)
    feetLP8/16      low-pass cutoff used for foot detection (published 12 Hz)
    SG30/60ms       Savitzky-Golay window of the adequacy diagnostic (published 40 ms)

For each configuration and channel it records the identifiable-only verdict
(identifiable_estimator.analyse: ok / weak / unusable), the fraction of beats
with accelerating decay, and the constraint spread across the five published
asymptote bounds (run_spread.py logic). The base configuration must reproduce
results_c4.csv for the same cases (printed as a consistency check).
"declined" = status 'unusable', "caveat" = 'weak', as in Table 15.

Cases: a fixed-seed random sample of the recordings that passed signal quality
in results_c4.csv.

Run in Colab:
    !python segmentation_sensitivity.py --c4 /content/drive/MyDrive/ppg_ident/results_c4.csv \
        --n 400 --out /content/drive/MyDrive/ppg_ident/results_segmentation.csv
Resumable.
"""

import argparse, os, time
import numpy as np
import pandas as pd
from scipy.signal import find_peaks, savgol_filter

from vitaldb_pipeline import load_waveforms, segment_quality, lowpass, FS
from tau_multibeat import tau_joint

WIN_S, SKIP_START_S, N_BEATS = 120.0, 600.0, 8
SEED = 2026
PRIORS = [(0.0, 10.0), (0.0, 25.0), (0.0, 40.0), (-20.0, 60.0), (-np.inf, np.inf)]

# Constants and logic of identifiable_estimator.py as used for results_c4.csv
# (desktop codes folder, 7 Aug, five-value split_window_test). Embedded so the
# run does not depend on which copy of that module is in the Colab session.
TAU_LO, TAU_HI = 0.15, 12.0
SPREAD_OK, SPLIT_TOL = 1.5, 1.30

BASE = dict(skip=0.25, shift_s=0.0, onset_lp=20.0, feet_lp=12.0, sg_s=0.04)
CONFIGS = {
    "base": {},
    "skip0.15": dict(skip=0.15), "skip0.35": dict(skip=0.35),
    "shift-20ms": dict(shift_s=-0.02), "shift+20ms": dict(shift_s=0.02),
    "onsetLP15": dict(onset_lp=15.0), "onsetLP25": dict(onset_lp=25.0),
    "feetLP8": dict(feet_lp=8.0), "feetLP16": dict(feet_lp=16.0),
    "SG30ms": dict(sg_s=0.03), "SG60ms": dict(sg_s=0.06),
}


# ---- parametrised copies of the published functions ----------------------
def detect_feet_p(x, fs=FS, feet_lp=12.0, min_hr=35.0, max_hr=180.0):
    """vitaldb_pipeline.detect_feet with the low-pass cutoff exposed."""
    xf = lowpass(x, fs, hi=feet_lp)
    idx, _ = find_peaks(-xf, distance=int(fs * 60.0 / max_hr), prominence=0.15 * np.ptp(xf))
    if idx.size < 3:
        return np.array([], dtype=int)
    rr = np.diff(idx) / fs
    keep = np.r_[True, (rr > 60.0 / max_hr) & (rr < 60.0 / min_hr)]
    return idx[keep]


def diastole_start_p(seg, fs=FS, skip=0.25, onset_lp=20.0, shift_s=0.0):
    """tau_fix.diastole_start with skip fraction, cutoff and an onset shift exposed."""
    n = len(seg)
    s = lowpass(seg, fs, hi=onset_lp)
    pk = int(np.argmax(s))
    if n - pk < int(0.15 * fs):
        return None
    d = np.gradient(s)
    steep = pk + int(np.argmin(d[pk:]))
    if n - steep < int(0.10 * fs):
        return None
    mins, _ = find_peaks(-s[steep:], distance=int(0.02 * fs))
    notch = steep + int(mins[0]) if mins.size else steep + int(0.05 * fs)
    start = notch + int(skip * (n - notch)) + int(round(shift_s * fs))
    if start <= notch or n - start < int(0.12 * fs):
        return None
    return int(start)


def collect_tails_p(sig, cfg, fs=FS, n_beats=N_BEATS, min_pts=40):
    """tau_multibeat._collect_tails with the parametrised feet and onset."""
    feet = detect_feet_p(sig, fs, cfg["feet_lp"])
    tails = []
    for a, b in zip(feet[:-1], feet[1:]):
        if len(tails) >= n_beats:
            break
        beat = np.asarray(sig[a:b], dtype=float)
        i0 = diastole_start_p(beat, fs, cfg["skip"], cfg["onset_lp"], cfg["shift_s"])
        if i0 is None:
            continue
        if beat.size - i0 >= min_pts:
            tails.append(beat[i0:])
    return tails, feet


def make_split_test(sg_s):
    """identifiable_estimator.split_window_test with the smoothing window exposed."""
    def split_window_test(tails, fs=FS, min_pts=60):
        se, sl, n_accel = [], [], 0
        for tl in tails:
            if tl.size < min_pts:
                continue
            w = min(int(sg_s * fs) | 1, (tl.size // 2) * 2 - 1)
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
            beat_ok, vals = True, []
            for seg_t, seg_l in ((tt[:h], ln[:h]), (tt[h:], ln[h:])):
                A = np.vstack([seg_t, np.ones_like(seg_t)]).T
                c, *_ = np.linalg.lstsq(A, seg_l, rcond=None)
                if c[0] >= 0:
                    beat_ok = False
                    break
                vals.append(-1.0 / c[0])
            if beat_ok:
                se.append(vals[0]); sl.append(vals[1])
            else:
                n_accel += 1
        n_tot = n_accel + len(se)
        frac = n_accel / n_tot if n_tot else np.nan
        if len(se) < 2 or len(sl) < 2:
            return np.nan, np.nan, np.nan, False, frac
        te, tl_ = float(np.median(se)), float(np.median(sl))
        if not (np.isfinite(te) and np.isfinite(tl_)) or te <= 0:
            return te, tl_, np.nan, False, frac
        ratio = tl_ / te
        ok = (n_accel == 0) and (1.0 / SPLIT_TOL <= ratio <= SPLIT_TOL)
        return te, tl_, float(ratio), bool(ok), float(frac)
    return split_window_test


def _fit_tails(tails, fs=FS, pinf_bounds=(-np.inf, np.inf), tau0=1.2):
    """identifiable_estimator._fit_tails, verbatim."""
    from scipy.optimize import least_squares
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
        r = least_squares(res, th0, bounds=(lo, hi), xtol=1e-12, ftol=1e-12, max_nfev=4000)
    except Exception:
        return np.nan, np.nan, np.nan, False
    tau, pinf = float(np.exp(r.x[0])), float(r.x[1])
    y = np.concatenate(tails)
    ss = float(np.sum(r.fun ** 2)); st = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss / st if st > 0 else np.nan
    railed = (abs(tau - TAU_LO) < 1e-3) or (abs(tau - TAU_HI) < 1e-3)
    return tau, r2, pinf, railed


def analyse_p(tails, split_test, fs=FS):
    """Verdict logic of identifiable_estimator.analyse (ok / weak / unusable)."""
    out = dict(status="unusable", model_adequate=False, frac_accelerating=np.nan)
    if len(tails) < 3:
        return out
    te, tl, ratio, adequate, frac = split_test(tails, fs)
    out.update(model_adequate=adequate, frac_accelerating=frac)
    if not adequate:
        return out
    taus, tau_free = [], np.nan
    for b in PRIORS:
        t, r2, _, railed = _fit_tails(tails, fs, pinf_bounds=b)
        if np.isfinite(t) and not railed:
            taus.append(t)
        if b[0] == -np.inf:
            tau_free = t
    if len(taus) < 2 or not np.isfinite(tau_free):
        return out
    out["status"] = "ok" if max(taus) / min(taus) <= SPREAD_OK else "weak"
    return out


def analyse_cfg(sig, cfg, with_spread=True):
    tails, feet = collect_tails_p(sig, cfg)
    r = analyse_p(tails, make_split_test(cfg["sg_s"]))
    out = dict(status=r["status"], adequate=r["model_adequate"],
               accel=r.get("frac_accelerating", np.nan), n_tails=len(tails))
    # constraint spread, run_spread.py logic (requires >= 4 tails)
    taus = []
    if with_spread and len(tails) >= 4:
        for b in PRIORS:
            tau, *_ = tau_joint(tails, FS, pinf_bounds=b)
            if np.isfinite(tau):
                taus.append(tau)
    out["spread"] = max(taus) / min(taus) if len(taus) > 1 and min(taus) > 0 else np.nan
    return out




def process_one(caseid):
    rec = dict(caseid=int(caseid), loaded=False, reason="")
    try:
        v = load_waveforms(caseid, fs=FS)
    except Exception as e:
        rec["reason"] = f"load:{type(e).__name__}"
        return rec
    if v is None or v.shape[0] < FS * (SKIP_START_S + WIN_S):
        rec["reason"] = "too short"
        return rec
    mid, half = v.shape[0] // 2, int(FS * WIN_S / 2)
    abp, ppg = v[mid - half:mid + half, 0], v[mid - half:mid + half, 1]
    ok, reason = segment_quality(abp, ppg, FS)
    if not ok:
        rec["reason"] = reason
        return rec
    abp = np.nan_to_num(abp, nan=float(np.nanmedian(abp)))
    ppg = np.nan_to_num(ppg, nan=float(np.nanmedian(ppg)))
    rec["loaded"] = True
    for name, delta in CONFIGS.items():
        cfg = {**BASE, **delta}
        for sig, pre in ((abp, "abp"), (ppg, "ppg")):
            # the smoothing window affects only the adequacy test, so the
            # constraint spread is not recomputed for the SG configurations
            r = analyse_cfg(sig, cfg, with_spread=not name.startswith("SG"))
            rec.update({f"{name}|{pre}_{k}": x for k, x in r.items()})
    return rec


# --------------------------------------------------------------------------
def q(s):
    s = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    return "n/a" if s.empty else f"{s.median():.2f} [{s.quantile(.25):.2f}, {s.quantile(.75):.2f}]"


def wilson(k, n, z=1.96):
    if n == 0:
        return np.nan, np.nan
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * (c - h), 100 * (c + h)


def report(df, c4_csv=None):
    ld = df[df.loaded == True]
    n = len(ld)
    print("\n" + "=" * 92)
    print(f"SEGMENTATION AND PREPROCESSING SENSITIVITY  (n = {n} recordings, seed {SEED})")
    print("=" * 92)
    if c4_csv and os.path.exists(c4_csv):
        c4 = pd.read_csv(c4_csv).set_index("caseid")
        m = ld.set_index("caseid").join(c4[["abp_status", "ppg_status"]], how="inner")
        for pre in ("abp", "ppg"):
            same = (m[f"base|{pre}_status"] == m[f"{pre}_status"]).mean()
            print(f"consistency, base vs results_c4.csv, {pre}: status identical in "
                  f"{100*same:.1f}% of {len(m)} cases (expected 100%)")
    for pre, lab in (("abp", "arterial line"), ("ppg", "plethysmogram")):
        print(f"\n{lab}")
        print(f"{'configuration':>14} {'declined % [95% CI]':>24} {'caveat %':>9} {'adequate %':>11} "
              f"{'accelerating beats %':>22} {'constraint spread':>20} {'spread<=1.5':>11}")
        for name in CONFIGS:
            st = ld[f"{name}|{pre}_status"]
            k = int((st == "unusable").sum())
            cav = 100 * (st == "weak").mean()
            lo, hi = wilson(k, n)
            adq = 100 * ld[f"{name}|{pre}_adequate"].astype(bool).mean()
            acc = pd.to_numeric(ld[f"{name}|{pre}_accel"], errors="coerce") * 100
            sp = pd.to_numeric(ld[f"{name}|{pre}_spread"], errors="coerce")
            le = 100 * np.mean(sp.dropna() <= 1.5) if sp.notna().any() else np.nan
            print(f"{name:>14} {f'{100*k/n:.1f} [{lo:.1f}, {hi:.1f}]':>24} {cav:9.1f} {adq:11.1f} "
                  f"{q(acc):>22} {(q(sp) if sp.notna().any() else 'as base'):>20} "
                  f"{(f'{le:10.1f}%' if np.isfinite(le) else '    as base')}")
        # paired change in accelerating fraction relative to base
        b = pd.to_numeric(ld[f"base|{pre}_accel"], errors="coerce")
        worst = max((abs(np.nanmedian(pd.to_numeric(ld[f"{c}|{pre}_accel"], errors="coerce") - b)), c)
                    for c in CONFIGS if c != "base")
        print(f"   largest median paired change in accelerating fraction vs base: "
              f"{100*worst[0]:.1f} points ({worst[1]})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--c4", required=True, help="results_c4.csv from the published run")
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--out", default="results_segmentation.csv")
    a = ap.parse_args()

    c4 = pd.read_csv(a.c4)
    pool = sorted(int(c) for c in c4.loc[c4.loaded == True, "caseid"])
    rng = np.random.default_rng(SEED)
    ids = sorted(rng.choice(pool, size=min(a.n, len(pool)), replace=False).tolist())
    print(f"cohort: {len(ids)} of {len(pool)} recordings that passed signal quality")

    rows, done = [], set()
    if os.path.exists(a.out):
        prev = pd.read_csv(a.out)
        rows, done = prev.to_dict("records"), set(prev.caseid)
        print(f"resuming: {len(done)} done")
    todo = [c for c in ids if c not in done]
    t0 = time.time()
    for i, cid in enumerate(todo, 1):
        rows.append(process_one(cid))
        if i % 10 == 0 or i == len(todo):
            pd.DataFrame(rows).to_csv(a.out, index=False)
            rate = i / max(time.time() - t0, 1e-9)
            print(f"  {i}/{len(todo)} {rate:.2f}/s ETA {(len(todo)-i)/max(rate,1e-9)/60:.1f} min",
                  flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(a.out, index=False)
    report(df, a.c4)
    print(f"\nsaved -> {a.out}")

"""
run_spread.py — primary clinical analysis.

Question
--------
Is the Windkessel time constant tau identifiable from real peripheral
waveforms, or is a reported tau determined by the analyst's prior on the
diastolic asymptote?

Method
------
For each case, fit a shared-tau / shared-P_inf / per-beat-amplitude model to
8 consecutive diastolic tails, repeated under five priors on P_inf ranging
from tight and physiological to unbounded. The ratio

        spread = max(tau) / min(tau)   across priors

measures prior dependence. On synthetic WK2 waveforms spread = 1.00; any
substantial excess on clinical data indicates that the WK2 diastolic decay
model does not hold at the measurement site.

Both channels are analysed: the calibrated arterial line (the field's
reference standard) and the uncalibrated plethysmogram.

Outputs
-------
    results_spread.csv   per-case estimates under every prior
    printed tables for Results sections 4.8 and 4.9
"""

import argparse, os, time
import numpy as np
import pandas as pd

from vitaldb_pipeline import (find_cohorts, load_waveforms, segment_quality,
                              detect_feet, FS)
from tau_fix import diastole_start
from tau_multibeat import _collect_tails, tau_joint

# ---- pre-specified -------------------------------------------------------
WIN_S        = 120.0
SKIP_START_S = 600.0
N_BEATS      = 8
MIN_TAILS    = 4
PRIORS = [("tight",     (0.0, 10.0)),
          ("physio",    (0.0, 25.0)),
          ("wide",      (0.0, 40.0)),
          ("permissive", (-20.0, 60.0)),
          ("free",      (-np.inf, np.inf))]
SPREAD_OK = 1.5      # pre-declared: spread below this = data-determined


# --------------------------------------------------------------------------
def analyse_channel(sig, fs=FS):
    """Fit under every prior. Returns dict of per-prior tau/pinf/r2 plus
    spread and a physiological-plausibility flag."""
    feet = detect_feet(sig, fs)
    if feet.size < MIN_TAILS + 1:
        return None
    tails = _collect_tails(sig, feet, diastole_start, fs, n_beats=N_BEATS)
    if len(tails) < MIN_TAILS:
        return None

    out, taus = {}, []
    for name, bounds in PRIORS:
        tau, r2, n, pinf, hit = tau_joint(tails, fs, pinf_bounds=bounds)
        out[f"tau_{name}"] = tau
        out[f"pinf_{name}"] = pinf
        out[f"r2_{name}"] = r2
        out[f"hit_{name}"] = hit
        if np.isfinite(tau):
            taus.append(tau)
    out["n_tails"] = len(tails)
    out["dias_s"] = float(np.mean([t.size for t in tails]) / fs)
    out["spread"] = (max(taus) / min(taus)) if len(taus) > 1 and min(taus) > 0 \
        else np.nan
    # physiological plausibility of the FREE fit, which has no prior imposed
    tf, pf = out.get("tau_free", np.nan), out.get("pinf_free", np.nan)
    out["free_plausible"] = bool(np.isfinite(tf) and 0.5 < tf < 4.0
                                 and np.isfinite(pf) and -10 < pf < 40)
    out["r2_free"] = out.get("r2_free", np.nan)
    return out


def process_one(caseid):
    rec = dict(caseid=int(caseid), ok=False, reason="")
    try:
        v = load_waveforms(caseid, fs=FS)
    except Exception as e:
        rec["reason"] = f"load:{type(e).__name__}"
        return rec
    if v is None or v.shape[0] < FS * (SKIP_START_S + WIN_S):
        rec["reason"] = "too short"
        return rec

    mid = v.shape[0] // 2
    half = int(FS * WIN_S / 2)
    abp, ppg = v[mid - half:mid + half, 0], v[mid - half:mid + half, 1]
    ok, reason = segment_quality(abp, ppg, FS)
    if not ok:
        rec["reason"] = reason
        return rec

    abp = np.nan_to_num(abp, nan=float(np.nanmedian(abp)))
    ppg = np.nan_to_num(ppg, nan=float(np.nanmedian(ppg)))

    a = analyse_channel(abp)
    p = analyse_channel(ppg)
    if a is None or p is None:
        rec["reason"] = "insufficient beats"
        return rec

    rec.update(ok=True, reason="ok",
               sbp=float(np.percentile(abp, 99)),
               dbp=float(np.percentile(abp, 1)),
               map=float(np.mean(abp)),
               hr=float(60.0 * FS / np.median(np.diff(detect_feet(abp, FS)))))
    rec.update({f"abp_{k}": v_ for k, v_ in a.items()})
    rec.update({f"ppg_{k}": v_ for k, v_ in p.items()})
    return rec


# --------------------------------------------------------------------------
def run(ids, outdir, tag):
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, f"results_{tag}.csv")
    rows, done = [], set()
    if os.path.exists(path):
        prev = pd.read_csv(path)
        rows = prev.to_dict("records")
        done = set(prev.caseid)
        print(f"resuming: {len(done)} done")
    todo = [c for c in ids if int(c) not in done]
    t0 = time.time()
    for i, cid in enumerate(todo, 1):
        rows.append(process_one(cid))
        if i % 20 == 0 or i == len(todo):
            pd.DataFrame(rows).to_csv(path, index=False)
            el = time.time() - t0
            rate = i / max(el, 1e-9)
            nok = sum(bool(r.get("ok")) for r in rows)
            print(f"  {i}/{len(todo)} ok={nok} {rate:.2f}/s "
                  f"ETA {(len(todo)-i)/max(rate,1e-9)/60:.1f} min", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    return df


# --------------------------------------------------------------------------
def q(s):
    s = pd.to_numeric(s, errors="coerce").dropna()
    if s.empty:
        return "n/a"
    return (f"{s.median():.3f} [{s.quantile(.25):.3f}, {s.quantile(.75):.3f}]")


def report(df):
    tot = len(df)
    ok = df[df.ok == True]
    print("\n" + "=" * 76)
    print("SECTION 4.8 -- prior dependence of the estimated time constant")
    print("=" * 76)
    print(f"attempted {tot}, analysable {len(ok)} "
          f"({100*len(ok)/max(tot,1):.1f}%)\n")
    print("quality-gate rejections:")
    for r, n in df[df.ok != True].reason.value_counts().items():
        print(f"   {str(r):28s} {n:5d} ({100*n/max(tot,1):.1f}%)")
    if len(ok) < 10:
        print("\ntoo few cases")
        return

    print(f"\nmedian diastolic window: {q(ok.abp_dias_s)} s   "
          f"HR {q(ok.hr)} min-1")

    print("\nTable: tau (s) under each prior on P_inf, arterial line")
    print(f"{'prior':>12} {'P_inf range':>16} {'tau median [IQR]':>28} "
          f"{'r2':>18} {'on bound':>9}")
    for name, (lo, hi) in PRIORS:
        hit = pd.to_numeric(ok[f"abp_hit_{name}"].astype(float),
                            errors="coerce").mean()
        print(f"{name:>12} {f'({lo}, {hi})':>16} "
              f"{q(ok[f'abp_tau_{name}']):>28} "
              f"{q(ok[f'abp_r2_{name}']):>18} {100*hit:8.1f}%")

    print("\nTable: same, plethysmogram")
    for name, (lo, hi) in PRIORS:
        print(f"{name:>12} {f'({lo}, {hi})':>16} "
              f"{q(ok[f'ppg_tau_{name}']):>28} "
              f"{q(ok[f'ppg_r2_{name}']):>18}")

    print("\n" + "-" * 76)
    for lab, col in [("arterial line", "abp_spread"), ("plethysmogram", "ppg_spread")]:
        s = pd.to_numeric(ok[col], errors="coerce").dropna()
        if s.empty:
            continue
        print(f"{lab:>16}: spread {q(s)}   "
              f"fraction <= {SPREAD_OK}x: {100*np.mean(s <= SPREAD_OK):.1f}%   "
              f"max {s.max():.1f}x")

    print("\nUnconstrained fits: quality versus plausibility")
    for lab, pre in [("arterial line", "abp"), ("plethysmogram", "ppg")]:
        r2 = pd.to_numeric(ok[f"{pre}_r2_free"], errors="coerce")
        pl = ok[f"{pre}_free_plausible"].astype(bool)
        print(f"   {lab:>14}: r2 {q(r2)}   "
              f"physiologically plausible {100*pl.mean():.1f}%   "
              f"r2>0.95 but implausible {100*((r2>0.95) & ~pl).mean():.1f}%")

    print("\n" + "=" * 76)
    print("SECTION 4.9 -- agreement between channels, per prior")
    print("=" * 76)
    print(f"{'prior':>12} {'n':>6} {'bias (s)':>10} {'LoA':>22} {'ICC':>7}")
    for name, _ in PRIORS:
        a = pd.to_numeric(ok[f"abp_tau_{name}"], errors="coerce")
        p = pd.to_numeric(ok[f"ppg_tau_{name}"], errors="coerce")
        m = np.isfinite(a) & np.isfinite(p)
        if m.sum() < 10:
            continue
        d = (p - a)[m]
        bias, sd = d.mean(), d.std(ddof=1)
        x, y = a[m].values, p[m].values
        M = np.column_stack([x, y]); n = len(x); gm = M.mean()
        msr = 2*np.sum((M.mean(1)-gm)**2)/(n-1)
        msc = n*np.sum((M.mean(0)-gm)**2)
        mse = np.sum((M - M.mean(1, keepdims=True) - M.mean(0, keepdims=True)
                      + gm)**2)/(n-1)
        icc = (msr-mse)/(msr+mse+2*(msc-mse)/n) if (msr+mse) else np.nan
        print(f"{name:>12} {m.sum():6d} {bias:+10.4f} "
              f"[{bias-1.96*sd:+.3f}, {bias+1.96*sd:+.3f}]".ljust(22) +
              f" {icc:7.3f}")
    print("\nIf the channel agreement itself changes with the prior, no")
    print("single agreement statistic characterises the comparison.")


# --------------------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--pilot", type=int, default=0)
    ap.add_argument("--outdir", default="./out")
    a = ap.parse_args()

    exp_a, _, _ = find_cohorts()
    ids = sorted(exp_a)
    if a.pilot:
        ids, tag = ids[:a.pilot], "spread_pilot"
    else:
        if a.n:
            ids = ids[:a.n]
        tag = "spread"
    df = run(ids, a.outdir, tag)
    report(df)
    print(f"\nsaved -> {os.path.join(a.outdir, f'results_{tag}.csv')}")
